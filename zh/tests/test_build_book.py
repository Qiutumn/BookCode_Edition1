from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import nbformat

from zh.tools import build_book, nb_tools


class BookSchemaTests(unittest.TestCase):
    def test_repository_manifest_orders_complete_edition(self) -> None:
        book, profiles, units = build_book.load_book(build_book.DEFAULT_BOOK)
        self.assertEqual(book["language"], "zh-CN")
        self.assertEqual(len(units), 18)
        self.assertEqual(
            [unit.id for unit in units[:5]],
            ["welcome", "dedication", "foreword", "preface", "symbol-list"],
        )
        self.assertEqual([unit.order for unit in units], sorted(unit.order for unit in units))
        chapter_three = next(unit for unit in units if unit.id == "chapter-3")
        self.assertEqual(
            chapter_three.manifest, build_book.ZH_ROOT / "manifests/chapter-3.toml"
        )
        self.assertEqual(chapter_three.environments, ("pymc", "tfp"))
        self.assertTrue({"static", "smoke", "release"}.issubset(profiles))

    def test_planned_unit_does_not_require_nonexistent_builder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book_path = root / "book.toml"
            book_path.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    [book]
                    title = "Fixture"
                    language = "zh-CN"
                    source_license = "CC-BY-NC-SA-4.0"
                    code_license = "GPL-2.0"
                    [profiles.static]
                    execute = false
                    [[units]]
                    id = "future-unit"
                    order = 1
                    kind = "chapter"
                    status = "planned"
                    """
                ),
                encoding="utf-8",
            )
            _, _, units = build_book.load_book(book_path)
            self.assertFalse(units[0].buildable)

    def test_invalid_schema_field_and_string_boolean_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_field = root / "field.toml"
            invalid_field.write_text(
                "schema_version = 1\nunknown = true\n[profiles.static]\nexecute = false\n",
                encoding="utf-8",
            )
            with self.assertRaises(build_book.ConfigError):
                build_book.load_book(invalid_field)

            string_bool = root / "bool.toml"
            string_bool.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    [book]
                    title = "Fixture"
                    language = "zh-CN"
                    source_license = "CC"
                    code_license = "GPL"
                    [profiles.static]
                    execute = "false"
                    [[units]]
                    id = "future-unit"
                    order = 1
                    kind = "chapter"
                    status = "planned"
                    """
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(build_book.ConfigError, "must be boolean"):
                build_book.load_book(string_bool)

    def test_solution_escape_and_external_builder_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for value in ("solutions/build.py", "answers/chapter.py", "../outside.py"):
                with self.subTest(value=value):
                    with self.assertRaises(build_book.ConfigError):
                        build_book.resolve_safe_path(base, value)

            unit = base / "unit"
            unit.mkdir()
            outside = base / "outside.py"
            outside.write_text("cells = []", encoding="utf-8")
            book = base / "book.toml"
            book.write_text(
                textwrap.dedent(
                    """
                    schema_version = 1
                    [book]
                    title = "Fixture"
                    language = "zh-CN"
                    source_license = "CC"
                    code_license = "GPL"
                    [profiles.static]
                    execute = false
                    [[units]]
                    id = "fixture"
                    order = 1
                    kind = "chapter"
                    status = "available"
                    path = "unit"
                    builder = "outside.py"
                    notebook = "fixture.ipynb"
                    org = "fixture.org"
                    """
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(build_book.ConfigError, "inside the unit"):
                build_book.load_book(book)

    def test_repository_smoke_matrix_resolves_composite_environment_locks(self) -> None:
        _, profiles, units = build_book.load_book(build_book.DEFAULT_BOOK)
        matrix = build_book.profile_matrix(
            profiles["smoke"], units, ["chapter-3"]
        )
        row = matrix["include"][0]
        self.assertEqual(row["environment"], "pymc+tfp")
        self.assertIn("pymc.lock.txt", row["locks"])
        self.assertIn("tfp.lock.txt", row["locks"])


class StaticCliTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        builder = root / "unit" / "build.py"
        builder.parent.mkdir()
        builder.write_text(
            textwrap.dedent(
                r"""
                from pathlib import Path

                # If imported from the live tree this would violate static-build
                # isolation. The orchestrator must import only a copied builder.
                Path(__file__).with_name("import-side-effect.txt").write_text(
                    "isolated", encoding="utf-8"
                )

                cells = [
                    {
                        "type": "markdown",
                        "source": "# Fixture",
                        "id": "fixture-title",
                        "metadata": {"kind": "translation", "provenance": ["fixture#title"]},
                    },
                    {
                        "type": "code",
                        "source": "from pathlib import Path\nPath('execution-side-effect.txt').write_text('workspace', encoding='utf-8')\nprint('executed')",
                        "id": "fixture-code",
                        "metadata": {"kind": "modernization", "provenance": ["fixture#code"]},
                    },
                ]
                if __name__ == "__main__":
                    raise RuntimeError("the orchestrator triggered __main__")
                """
            ),
            encoding="utf-8",
        )
        book = root / "book.toml"
        book.write_text(
            textwrap.dedent(
                """
                schema_version = 1
                [book]
                title = "Fixture"
                language = "zh-CN"
                source_license = "CC-BY-NC-SA-4.0"
                code_license = "GPL-2.0"

                [profiles.static]
                execute = false
                timeout = 30
                allow_errors = false
                units = ["fixture"]
                environment = "core"

                [profiles.smoke]
                execute = true
                timeout = 7
                allow_errors = false
                units = ["fixture"]
                environment = "core"

                [profiles.release]
                execute = true
                timeout = 10
                allow_errors = false
                units = ["fixture"]
                environment = "core"

                [[units]]
                id = "fixture"
                order = 1
                kind = "chapter"
                title = "Fixture"
                status = "available"
                path = "unit"
                builder = "unit/build.py"
                notebook = "fixture.ipynb"
                org = "fixture.org"
                assets_dir = "generated"
                timeout = 999
                """
            ),
            encoding="utf-8",
        )
        return book

    @staticmethod
    def _run(args: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = build_book.main(args)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_static_validation_imports_copy_without_running_main_or_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            result, stdout, stderr = self._run(["validate", "--book", str(book)])
            self.assertEqual(result, 0, stderr)
            self.assertIn("validated 1 units and 3 profiles", stdout)
            self.assertFalse((root / "unit/import-side-effect.txt").exists())

    def test_build_creates_fresh_attested_stage_and_check_does_not_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            stage = root / "stage"
            result, _, stderr = self._run(
                ["build", "--book", str(book), "--staging-dir", str(stage)]
            )
            self.assertEqual(result, 0, stderr)
            self.assertTrue((stage / "unit/fixture.ipynb").is_file())
            self.assertTrue((stage / "unit/fixture.org").is_file())
            self.assertTrue((stage / build_book.ATTESTATION_NAME).is_file())
            self.assertFalse((root / "unit/fixture.ipynb").exists())
            self.assertFalse((root / "unit/fixture.org").exists())
            self.assertFalse((root / "unit/import-side-effect.txt").exists())

            # A checked stage is independent of subsequent authored-builder text.
            (root / "unit/build.py").write_text("this is invalid python", encoding="utf-8")
            result, stdout, stderr = self._run(
                ["check", "--book", str(book), "--staging-dir", str(stage)]
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("checked 1 attested unit", stdout)

            marker = stage / "marker"
            marker.write_text("preserve", encoding="utf-8")
            result, _, stderr = self._run(
                ["build", "--book", str(book), "--staging-dir", str(stage)]
            )
            self.assertEqual(result, 2)
            self.assertIn("already exists", stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_executed_build_uses_workspace_and_profile_timeout_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            stage = root / "stage"
            observed: dict[str, object] = {}
            real_execute = build_book._execute_notebook

            def recording_execute(*args, **kwargs):
                observed.update(kwargs)
                return real_execute(*args, **kwargs)

            with patch.object(build_book, "_execute_notebook", side_effect=recording_execute):
                result, _, stderr = self._run(
                    [
                        "build",
                        "--book",
                        str(book),
                        "--profile",
                        "smoke",
                        "--staging-dir",
                        str(stage),
                    ]
                )
            self.assertEqual(result, 0, stderr)
            self.assertEqual(observed["timeout"], 7)
            self.assertEqual(observed["profile_name"], "smoke")
            self.assertFalse((root / "unit/execution-side-effect.txt").exists())
            self.assertFalse((root / "unit/import-side-effect.txt").exists())
            self.assertEqual(
                (stage / "unit/execution-side-effect.txt").read_text(encoding="utf-8"),
                "workspace",
            )
            notebook = nb_tools.validate_notebook(
                stage / "unit/fixture.ipynb", require_executed=True
            )
            self.assertEqual(notebook.cells[1].execution_count, 1)

    def test_attestation_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            stage = root / "stage"
            result, _, stderr = self._run(
                ["build", "--book", str(book), "--staging-dir", str(stage)]
            )
            self.assertEqual(result, 0, stderr)
            with (stage / "unit/fixture.org").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            result, _, stderr = self._run(
                ["check", "--book", str(book), "--staging-dir", str(stage)]
            )
            self.assertEqual(result, 2)
            self.assertIn("changed after attestation", stderr)

    def test_command_separation_and_release_execution_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            result, _, stderr = self._run(
                [
                    "release",
                    "--book",
                    str(book),
                    "--staging-dir",
                    str(root / "release-no-execute"),
                    "--no-execute",
                ]
            )
            self.assertEqual(result, 2)
            self.assertIn("cannot use --no-execute", stderr)

            stage = root / "stage"
            result, _, stderr = self._run(
                ["build", "--book", str(book), "--staging-dir", str(stage)]
            )
            self.assertEqual(result, 0, stderr)
            result, _, stderr = self._run(
                [
                    "check",
                    "--book",
                    str(book),
                    "--staging-dir",
                    str(stage),
                    "--promote",
                ]
            )
            self.assertEqual(result, 2)
            self.assertIn("check never promotes", stderr)
            self.assertFalse((root / "unit/fixture.ipynb").exists())

    def test_release_can_promote_only_after_attested_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            result, stdout, stderr = self._run(
                [
                    "release",
                    "--book",
                    str(book),
                    "--staging-dir",
                    str(root / "release-stage"),
                    "--promote",
                ]
            )
            self.assertEqual(result, 0, stderr)
            self.assertIn("promoted", stdout)
            self.assertTrue((root / "unit/fixture.ipynb").is_file())
            self.assertTrue((root / "unit/fixture.org").is_file())
            self.assertTrue((root / "unit/execution-side-effect.txt").is_file())
            self.assertFalse((root / "unit/import-side-effect.txt").exists())

    def test_stale_managed_output_refuses_promotion_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            generated = root / "unit/generated"
            generated.mkdir()
            stale = generated / "stale.svg"
            stale.write_text("old", encoding="utf-8")
            result, _, stderr = self._run(
                [
                    "release",
                    "--book",
                    str(book),
                    "--staging-dir",
                    str(root / "release-stage"),
                    "--promote",
                ]
            )
            self.assertEqual(result, 2)
            self.assertIn(str(stale), stderr)
            self.assertIn("explicit pruning authorization", stderr)
            self.assertEqual(stale.read_text(encoding="utf-8"), "old")
            self.assertFalse((root / "unit/fixture.ipynb").exists())
            self.assertFalse((root / "unit/fixture.org").exists())

    def test_stage_path_may_not_exist_or_overlap_live_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            book = self._fixture(root)
            _, profiles, units = build_book.load_book(book)
            live_child = root / "unit/new-stage"
            with self.assertRaisesRegex(build_book.ConfigError, "overlaps live"):
                build_book._make_stage(live_child, units, root)
            self.assertFalse(live_child.exists())

            existing = root / "existing"
            existing.mkdir()
            marker = existing / "marker"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(build_book.ConfigError, "already exists"):
                build_book._make_stage(existing, units, root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_atomic_promotion_rolls_back_all_destinations_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "unit"
            source_dir.mkdir()
            builder = source_dir / "build.py"
            builder.write_text("cells = []", encoding="utf-8")
            unit = build_book.Unit(
                id="fixture",
                order=1,
                kind="chapter",
                title="Fixture",
                status="available",
                path=Path("unit"),
                root=root,
                builder=builder,
                notebook="one.txt",
                org="two.txt",
            )
            stage = root / "stage"
            staged_dir = stage / "unit"
            staged_dir.mkdir(parents=True)
            first_source = staged_dir / "one.txt"
            second_source = staged_dir / "two.txt"
            first_source.write_text("new-one", encoding="utf-8")
            second_source.write_text("new-two", encoding="utf-8")
            first_destination = source_dir / "one.txt"
            second_destination = source_dir / "two.txt"
            first_destination.write_text("old-one", encoding="utf-8")
            second_destination.write_text("old-two", encoding="utf-8")
            result = build_book.BuildResult(
                unit=unit, files=[first_source, second_source]
            )

            real_replace = os.replace
            installs = 0

            def fail_second_install(source, destination):
                nonlocal installs
                if ".new-" in Path(source).name:
                    installs += 1
                    if installs == 2:
                        raise OSError("simulated install failure")
                return real_replace(source, destination)

            with patch.object(build_book.os, "replace", side_effect=fail_second_install):
                with self.assertRaisesRegex(OSError, "simulated"):
                    build_book.atomic_promote([result], stage)
            self.assertEqual(first_destination.read_text(encoding="utf-8"), "old-one")
            self.assertEqual(second_destination.read_text(encoding="utf-8"), "old-two")
            self.assertFalse(list(source_dir.glob(".*.new-*")))
            self.assertFalse(list(source_dir.glob(".*.backup-*")))


if __name__ == "__main__":
    unittest.main()
