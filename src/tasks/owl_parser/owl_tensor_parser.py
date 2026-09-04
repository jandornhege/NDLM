"""Convert RDF/XML OWL ontologies into NDLM concept and role tensors."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import torch
from rdflib import Graph, RDF
from rdflib.compare import to_canonical_graph
from rdflib.term import Node, URIRef


PathInput = Union[str, Path]


def rdf_term_id(term: Node) -> str:
    """Return an unambiguous, RDF-compatible identifier for a graph term."""
    return term.n3()


@dataclass(frozen=True)
class OwlTensorData:
    """A tensor encoding of an RDF graph over retained entity objects.

    ``concepts[c, o]`` is one exactly when object ``o`` has RDF type ``c``.
    ``roles[r, s, o]`` is one exactly when role ``r`` relates subject ``s`` to
    object ``o``. By default only IRI terms are retained as objects, excluding
    literals such as labels and descriptions plus blank-node RDF syntax.
    """

    concepts: torch.Tensor
    roles: torch.Tensor
    object_terms: tuple[str, ...]
    concept_names: tuple[str, ...]
    role_names: tuple[str, ...]
    object_index: dict[str, int]
    concept_index: dict[str, int]
    role_index: dict[str, int]

def local_neighborhood(
    graph: Graph,
    start: Node,
    distance: int = 4,
) -> set[Node]:
    """Return all nodes within `distance` RDF edges of `start`."""

    visited = {start}
    frontier = {start}

    for _ in range(distance):
        next_frontier = set()

        for node in frontier:
            # Outgoing edges: node -> object
            for _, predicate, obj in graph.triples((node, None, None)):
                if predicate == RDF.type:
                    continue
                if obj not in visited:
                    next_frontier.add(obj)

            # Incoming edges: subject -> node
            for subject, predicate, _ in graph.triples((None, None, node)):
                if predicate == RDF.type:
                    continue

                if subject not in visited:
                    next_frontier.add(subject)

        visited.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    return visited


@lru_cache(maxsize=8)
def _load_canonical_graph(paths: tuple[Path, ...]) -> Graph:
    """Parse and canonicalize an OWL graph once and reuse it across calls.

    `owl_to_tensors` is invoked once per example, but the underlying
    ontology is identical across examples, so parsing and canonicalizing
    it from disk every time is redundant. The returned graph is only
    ever read from (never mutated) by callers.
    """

    graph = Graph()

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

        suffix = path.suffix.lower()

        if suffix == ".ttl":
            graph.parse(path, format="turtle")
        elif suffix == ".txt":
            graph.parse(path, format="nt")
        else:
            graph.parse(path)
    return to_canonical_graph(graph)


def owl_to_tensors(
    owl_files: Union[PathInput, Iterable[PathInput]],
    *,
    padding: int = 0,
    export_path: PathInput | None = None,
    max_role_tensor_bytes: int | None = None,
    include_non_iri_terms: bool = False,
    neighborhood_node: Node | str | None = None,
    neighborhood_distance: int = 4,
) -> OwlTensorData:
    """Parse RDF/XML OWL files into binary concept vectors and role matrices.

    If ``neighborhood_node`` is supplied, only objects within
    ``neighborhood_distance`` role edges of that node are included.

    The concept and role vocabularies are always computed from the
    complete graph before neighborhood filtering. Therefore, their
    ordering and dimensionality remain fixed across local neighborhoods.
    """

    if padding < 0:
        raise ValueError("padding must be non-negative")

    if max_role_tensor_bytes is not None and max_role_tensor_bytes < 0:
        raise ValueError("max_role_tensor_bytes must be non-negative")

    if neighborhood_distance < 0:
        raise ValueError("neighborhood_distance must be non-negative")

    # ---------------------------------------------------------
    # Resolve input paths
    # ---------------------------------------------------------

    if isinstance(owl_files, (str, Path)):
        paths = (Path(owl_files),)
    else:
        paths = tuple(Path(owl_file) for owl_file in owl_files)

    if not paths:
        raise ValueError("at least one OWL file is required")

    # ---------------------------------------------------------
    # Load complete RDF graph
    # ---------------------------------------------------------

    graph = _load_canonical_graph(paths)

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    def keep_object(term: Node) -> bool:
        return (
            include_non_iri_terms
            or isinstance(term, URIRef)
        )

    # ---------------------------------------------------------
    # GLOBAL concept / role vocabulary
    #
    # IMPORTANT:
    # These are computed BEFORE local-neighborhood filtering.
    # Thus every local graph has the same concept and role
    # dimensions and the same ordering.
    # ---------------------------------------------------------

    concept_terms = {
        obj
        for subject, predicate, obj in graph
        if predicate == RDF.type
        and keep_object(subject)
        and keep_object(obj)
    }

    role_terms = {
        predicate
        for subject, predicate, obj in graph
        if predicate != RDF.type
        and keep_object(subject)
        and keep_object(obj)
    }

    ordered_concepts = tuple(
        sorted(concept_terms, key=rdf_term_id)
    )

    ordered_roles = tuple(
        sorted(role_terms, key=rdf_term_id)
    )

    concept_names = tuple(
        rdf_term_id(term)
        for term in ordered_concepts
    )

    role_names = tuple(
        rdf_term_id(term)
        for term in ordered_roles
    )

    # ---------------------------------------------------------
    # Filter graph to local neighborhood
    # ---------------------------------------------------------

    if neighborhood_node is not None:

        if isinstance(neighborhood_node, str):
            neighborhood_node = URIRef(
                neighborhood_node.strip("<>")
            )

        if neighborhood_node not in graph.all_nodes():
            raise ValueError(
                "neighborhood node not found in graph: "
                f"{neighborhood_node}"
            )

        neighborhood = local_neighborhood(
            graph,
            neighborhood_node,
            distance=neighborhood_distance,
        )

        filtered_graph = Graph()

        for subject, predicate, obj in graph:

            # -------------------------------------------------
            # Role edges:
            # Keep only edges completely inside neighborhood.
            # -------------------------------------------------

            if predicate != RDF.type:
                if (
                    subject in neighborhood
                    and obj in neighborhood
                ):
                    filtered_graph.add(
                        (subject, predicate, obj)
                    )

            # -------------------------------------------------
            # rdf:type:
            #
            # Keep type assertions for local individuals even
            # if the class itself is not in the neighborhood.
            # -------------------------------------------------

            else:
                if subject in neighborhood:
                    filtered_graph.add(
                        (subject, predicate, obj)
                    )

        graph = filtered_graph

    # ---------------------------------------------------------
    # Determine LOCAL object vocabulary
    #
    # Unlike concepts and roles, objects are allowed to differ
    # between neighborhoods.
    # ---------------------------------------------------------

    terms = {
        term
        for subject, _, obj in graph
        for term in (subject, obj)
        if keep_object(term)
    }

    ordered_terms = tuple(
        sorted(terms, key=rdf_term_id)
    )

    object_terms = tuple(
        rdf_term_id(term)
        for term in ordered_terms
    )

    # ---------------------------------------------------------
    # Indices
    # ---------------------------------------------------------

    object_index = {
        term: index
        for index, term in enumerate(object_terms)
    }

    concept_index = {
        name: index
        for index, name in enumerate(concept_names)
    }

    role_index = {
        name: index
        for index, name in enumerate(role_names)
    }

    # ---------------------------------------------------------
    # Tensor dimensions
    # ---------------------------------------------------------

    num_objects = len(ordered_terms) + padding

    role_tensor_bytes = (
        len(ordered_roles)
        * num_objects
        * num_objects
        * 4
    )

    if (
        max_role_tensor_bytes is not None
        and role_tensor_bytes > max_role_tensor_bytes
    ):
        raise MemoryError(
            "dense role tensor requires "
            f"{role_tensor_bytes / 2**30:.2f} GiB, "
            "exceeding the "
            f"{max_role_tensor_bytes / 2**30:.2f} GiB limit"
        )

    # ---------------------------------------------------------
    # Allocate tensors
    #
    # IMPORTANT:
    # len(ordered_concepts) and len(ordered_roles) are GLOBAL.
    # Therefore these dimensions remain fixed.
    # ---------------------------------------------------------

    concepts = torch.zeros(
        (len(ordered_concepts), num_objects),
        dtype=torch.float32,
    )

    roles = torch.zeros(
        (len(ordered_roles), num_objects, num_objects),
        dtype=torch.float32,
    )

    # ---------------------------------------------------------
    # Fill tensors
    # ---------------------------------------------------------

    for subject, predicate, obj in graph:

        subject_id = rdf_term_id(subject)
        object_id = rdf_term_id(obj)

        if (
            subject_id not in object_index
            or object_id not in object_index
        ):
            continue

        subject_index = object_index[subject_id]
        object_index_value = object_index[object_id]

        if predicate == RDF.type:

            concept_id = rdf_term_id(obj)

            # Should always exist because concept_names came
            # from the complete graph.
            concepts[
                concept_index[concept_id],
                subject_index,
            ] = 1.0

        else:

            role_id = rdf_term_id(predicate)

            # Should always exist because role_names came
            # from the complete graph.
            roles[
                role_index[role_id],
                subject_index,
                object_index_value,
            ] = 1.0

    # ---------------------------------------------------------
    # Package result
    # ---------------------------------------------------------

    tensor_data = OwlTensorData(
        concepts=concepts,
        roles=roles,
        object_terms=object_terms,
        concept_names=concept_names,
        role_names=role_names,
        object_index=object_index,
        concept_index=concept_index,
        role_index=role_index,
    )

    # ---------------------------------------------------------
    # Optional export
    # ---------------------------------------------------------

    if export_path is not None:
        output_path = Path(export_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "concepts": tensor_data.concepts,
                "roles": tensor_data.roles,
                "object_terms": tensor_data.object_terms,
                "concept_names": tensor_data.concept_names,
                "role_names": tensor_data.role_names,
                "object_index": tensor_data.object_index,
                "concept_index": tensor_data.concept_index,
                "role_index": tensor_data.role_index,
            },
            output_path,
        )

    return tensor_data