from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from tasks.owl_parser.owl_tensor_parser import owl_to_tensors, rdf_term_id
from rdflib import Graph, Namespace, RDF
from rdflib.compare import to_canonical_graph


OWL_PARSER_DIR = Path(__file__).parents[1] / "src" / "tasks" / "owl_parser"
NTN = Namespace("http://semanticbible.org/ns/2006/NTNames#")


class OwlTensorParserTests(unittest.TestCase):
    @staticmethod
    def _ontology_paths():
        return [
            OWL_PARSER_DIR / "NTNames.owl.xml",
            OWL_PARSER_DIR / "NTN-individuals.owl.xml",
        ]

    def test_supplied_ontologies_preserve_types_resources_and_literals(self):
        data = owl_to_tensors(self._ontology_paths(), padding=2)

        salem_index = data.object_index[rdf_term_id(NTN.Salem)]
        city_index = data.concept_index[rdf_term_id(NTN.City)]
        location_index = data.role_index[rdf_term_id(NTN.location)]
        geodata_index = data.object_index[rdf_term_id(NTN.SalemGeodata)]

        print(f"Concepts ({len(data.concept_names)}): {list(data.concept_names)}")
        print(f"Roles ({len(data.role_names)}): {list(data.role_names)}")

        self.assertEqual(data.concepts.dtype, data.roles.dtype)
        self.assertEqual(data.concepts.dtype, __import__("torch").float32)
        self.assertEqual(data.concepts[city_index, salem_index].item(), 1.0)
        self.assertEqual(data.roles[location_index, salem_index, geodata_index].item(), 1.0)
        self.assertEqual(data.concepts.shape[1], len(data.object_terms) + 2)
        self.assertEqual(data.roles.shape[1:], (len(data.object_terms) + 2,) * 2)
        self.assertTrue(any("31.77451070780678" in term for term in data.object_terms))
        self.assertTrue(set(data.concepts.unique().tolist()).issubset({0.0, 1.0}))
        self.assertTrue(set(data.roles.unique().tolist()).issubset({0.0, 1.0}))

    def test_export_contains_tensors_and_metadata(self):
        with TemporaryDirectory() as directory:
            export_path = Path(directory) / "nested" / "ntn_tensors.pt"
            data = owl_to_tensors(
                self._ontology_paths(),
                export_path=export_path,
            )
            exported = torch.load(export_path, weights_only=True)

        self.assertTrue(torch.equal(exported["concepts"], data.concepts))
        self.assertTrue(torch.equal(exported["roles"], data.roles))
        self.assertEqual(exported["concept_names"], data.concept_names)
        self.assertEqual(exported["role_names"], data.role_names)
        self.assertEqual(exported["object_terms"], data.object_terms)

    def test_every_normalized_rdf_triple_has_a_tensor_encoding(self):
        graph = Graph()
        for path in self._ontology_paths():
            graph.parse(path, format="xml")
        graph = to_canonical_graph(graph)

        data = owl_to_tensors(self._ontology_paths())
        for subject, predicate, obj in graph:
            subject_index = data.object_index[rdf_term_id(subject)]
            object_index = data.object_index[rdf_term_id(obj)]
            if predicate == RDF.type:
                concept_index = data.concept_index[rdf_term_id(obj)]
                self.assertEqual(data.concepts[concept_index, subject_index].item(), 1.0)
            else:
                role_index = data.role_index[rdf_term_id(predicate)]
                self.assertEqual(data.roles[role_index, subject_index, object_index].item(), 1.0)


if __name__ == "__main__":
    unittest.main()