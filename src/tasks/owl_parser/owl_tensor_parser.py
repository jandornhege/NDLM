"""Convert RDF/XML OWL ontologies into NDLM concept and role tensors."""

from collections.abc import Iterable
from dataclasses import dataclass
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


def owl_to_tensors(
    owl_files: Union[PathInput, Iterable[PathInput]],
    *,
    padding: int = 0,
    export_path: PathInput | None = None,
    max_role_tensor_bytes: int | None = None,
    include_non_iri_terms: bool = False,
) -> OwlTensorData:
    """Parse RDF/XML OWL files into binary concept vectors and role matrices.

    Pass every ontology file that should be part of the same graph, for example
    both ``NTNames.owl.xml`` and ``NTN-individuals.owl.xml``. RDF imports are
    not fetched from the network; this keeps parsing deterministic and lets the
    caller choose exactly which local ontologies are included. By default,
    literals and blank nodes are excluded from the object dimension; set
    ``include_non_iri_terms`` to retain the lossless RDF-term encoding. When
    ``export_path`` is supplied, the tensors and their name/index metadata are
    written together with ``torch.save``. ``max_role_tensor_bytes`` rejects a
    dense role tensor before allocating it.
    """
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_role_tensor_bytes is not None and max_role_tensor_bytes < 0:
        raise ValueError("max_role_tensor_bytes must be non-negative")

    if isinstance(owl_files, (str, Path)):
        paths = (Path(owl_files),)
    else:
        paths = tuple(Path(owl_file) for owl_file in owl_files)

    if not paths:
        raise ValueError("at least one OWL file is required")

    graph = Graph()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        graph.parse(path, format="xml")

    graph = to_canonical_graph(graph)
    def keep_object(term: Node) -> bool:
        return include_non_iri_terms or isinstance(term, URIRef)

    terms = {
        term
        for subject, _, obj in graph
        for term in (subject, obj)
        if keep_object(term)
    }
    concept_terms = {
        obj
        for subject, predicate, obj in graph
        if predicate == RDF.type and keep_object(subject) and keep_object(obj)
    }
    role_terms = {
        predicate
        for subject, predicate, obj in graph
        if predicate != RDF.type and keep_object(subject) and keep_object(obj)
    }

    ordered_terms = tuple(sorted(terms, key=rdf_term_id))
    ordered_concepts = tuple(sorted(concept_terms, key=rdf_term_id))
    ordered_roles = tuple(sorted(role_terms, key=rdf_term_id))

    object_terms = tuple(rdf_term_id(term) for term in ordered_terms)
    concept_names = tuple(rdf_term_id(term) for term in ordered_concepts)
    role_names = tuple(rdf_term_id(term) for term in ordered_roles)
    object_index = {term: index for index, term in enumerate(object_terms)}
    concept_index = {term: index for index, term in enumerate(concept_names)}
    role_index = {term: index for index, term in enumerate(role_names)}

    num_objects = len(ordered_terms) + padding
    role_tensor_bytes = len(ordered_roles) * num_objects * num_objects * 4
    if (
        max_role_tensor_bytes is not None
        and role_tensor_bytes > max_role_tensor_bytes
    ):
        raise MemoryError(
            "dense role tensor requires "
            f"{role_tensor_bytes / 2**30:.2f} GiB, exceeding the "
            f"{max_role_tensor_bytes / 2**30:.2f} GiB limit"
        )
    concepts = torch.zeros((len(ordered_concepts), num_objects), dtype=torch.float32)
    roles = torch.zeros(
        (len(ordered_roles), num_objects, num_objects), dtype=torch.float32
    )

    for subject, predicate, obj in graph:
        subject_id = rdf_term_id(subject)
        object_id = rdf_term_id(obj)
        if subject_id not in object_index or object_id not in object_index:
            continue
        subject_index = object_index[subject_id]
        object_index_value = object_index[object_id]
        if predicate == RDF.type:
            concepts[concept_index[rdf_term_id(obj)], subject_index] = 1.0
        else:
            roles[
                role_index[rdf_term_id(predicate)], subject_index, object_index_value
            ] = 1.0

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

    if export_path is not None:
        output_path = Path(export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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