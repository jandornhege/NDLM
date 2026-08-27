"""Export every RDF/XML ontology below a directory as a colocated .pt payload."""

import argparse
from pathlib import Path

from tasks.owl_parser.owl_tensor_parser import owl_to_tensors


DEFAULT_INPUT_DIRECTORY = Path("src/data/Ontolearn")


def ontology_files(input_directory: Path) -> list[Path]:
    return sorted(
        path
        for suffix in ("*.owl", "*.xml")
        for path in input_directory.rglob(suffix)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument(
        "--output-suffix",
        default=".pt",
        help="Suffix appended after the ontology filename, such as '.pt'.",
    )
    parser.add_argument(
        "--only",
        type=Path,
        action="append",
        help="Ontology path relative to --input-directory; may be repeated.",
    )
    parser.add_argument(
        "--max-role-tensor-gib",
        type=float,
        default=8.0,
        help="Skip an ontology when its dense role tensor exceeds this size.",
    )
    parser.add_argument(
        "--include-non-iri-terms",
        action="store_true",
        help="Retain RDF literals and blank nodes as objects instead of filtering them.",
    )
    args = parser.parse_args()

    input_directory = args.input_directory.resolve()
    if not input_directory.is_dir():
        parser.error(f"input directory does not exist: {input_directory}")

    if args.only:
        inputs = [input_directory / relative_path for relative_path in args.only]
    else:
        inputs = ontology_files(input_directory)
    if not inputs:
        parser.error("no OWL or XML ontology files were found")
    if args.max_role_tensor_gib < 0:
        parser.error("--max-role-tensor-gib must be non-negative")

    max_role_tensor_bytes = int(args.max_role_tensor_gib * 2**30)
    exported = []
    skipped = []

    for input_path in inputs:
        if not input_path.is_file():
            parser.error(f"ontology does not exist: {input_path}")
        output_path = input_path.with_name(input_path.name + args.output_suffix)
        try:
            tensor_data = owl_to_tensors(
                input_path,
                export_path=output_path,
                max_role_tensor_bytes=max_role_tensor_bytes,
                include_non_iri_terms=args.include_non_iri_terms,
            )
        except MemoryError as error:
            skipped.append((input_path, str(error)))
            print(f"skipped {input_path.relative_to(input_directory)}: {error}")
            continue
        exported.append((input_path, tensor_data))
        print(
            f"exported {input_path.relative_to(input_directory)} -> "
            f"{output_path.relative_to(input_directory)} "
            f"(objects={len(tensor_data.object_terms)}, "
            f"concepts={len(tensor_data.concept_names)}, "
            f"roles={len(tensor_data.role_names)})"
        )

    print(f"completed: {len(exported)} exported, {len(skipped)} skipped")


if __name__ == "__main__":
    main()