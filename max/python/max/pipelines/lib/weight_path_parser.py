# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""MAX config classes."""

from __future__ import annotations

import logging
from pathlib import Path

import huggingface_hub

logger = logging.getLogger("max.pipelines")


class WeightPathParser:
    """Parses and validates weight paths for model configuration."""

    @staticmethod
    def parse(
        model_path: str,
        weight_path: list[Path]
        | list[str]
        | tuple[Path, ...]
        | tuple[str, ...]
        | Path
        | str,
    ) -> tuple[list[Path], str | None]:
        """Parse weight paths and extract any weights repo ID.

        Args:
            model_path: The model path to use for parsing the weight path(s)
            weight_path: The weight path(s) to parse - can be a single path, tuple, or list

        Examples:
            >>> WeightPathParser.parse("org/model", "path/to/weights.safetensors")
            (["path/to/weights.safetensors"], None)

            >>> WeightPathParser.parse("org/model", ["path/to/weights1.safetensors", "path/to/weights2.safetensors"])
            (["path/to/weights1.safetensors", "path/to/weights2.safetensors"], None)

            >>> WeightPathParser.parse("org/model", Path("path/to/weights.safetensors"))
            ([Path("path/to/weights.safetensors")], None)

            >>> WeightPathParser.parse("org/model", "org/model/weights.safetensors")
            ([Path("weights.safetensors")], "org/model")

            >>> WeightPathParser.parse("org/model", ["local_weights.safetensors", "other_org/other_model/remote_weights.safetensors"])
            ([Path("local_weights.safetensors"), Path("remote_weights.safetensors")], "other_org/other_model")

            # This is a special case where the weight path doesn't have a HF prefix,
            # which means file_exists will return False so treat the whole path
            # (without trimming a potential repo id prefix) as a local path.
            >>> WeightPathParser.parse("org/model", "very/nested/subfolder/another_nested/weights.safetensors")
            ([Path("very/nested/subfolder/another_nested/weights.safetensors")], None)

            # This on the other hand should fail, since model_path is empty
            # and we can't derive the repo id from the weight path.
            >>> WeightPathParser.parse("", "very/nested/subfolder/another_nested/weights.safetensors")
            Traceback (most recent call last):
            ...
            ValueError: Unable to derive model_path from weight_path, please provide a valid Hugging Face repository id.

        Returns:
            A tuple of (processed_weight_paths, weights_repo_id)

        Raises:
            ValueError: If weight paths are invalid or cannot be processed
        """
        # Normalize to list
        if isinstance(weight_path, tuple):
            weight_path_list = list(weight_path)
        elif not isinstance(weight_path, list):
            weight_path_list = [weight_path]
        else:
            weight_path_list = weight_path  # type: ignore

        weight_paths = []
        hf_weights_repo_id = None

        for path in weight_path_list:
            # Convert strings to Path objects and validate types
            if isinstance(path, str):
                path = Path(path)
            elif not isinstance(path, Path):
                raise ValueError(
                    "weight_path provided must either be string or Path:"
                    f" '{path}'"
                )

            # If path already exists as a file, add it directly
            if path.is_file():
                weight_paths.append(path)
                continue

            # Parse potential Hugging Face repo ID from path
            path, extracted_repo_id = (
                WeightPathParser._parse_huggingface_repo_path(model_path, path)
            )
            if extracted_repo_id:
                hf_weights_repo_id = extracted_repo_id

            weight_paths.append(path)

        return weight_paths, hf_weights_repo_id

    @staticmethod
    def _parse_huggingface_repo_path(
        model_path: str, path: Path
    ) -> tuple[Path, str | None]:
        """Parse a path that may contain a Hugging Face repo ID.

        Args:
            model_path: The model path to use for parsing the weight path(s)
            path: The local path to parse HF artifacts from

        Returns:
            A tuple of (processed_path, extracted_repo_id)

        Raises:
            ValueError: If unable to derive model_path from weight_path when needed
        """
        path_pieces = str(path).split("/")

        error_message = (
            "Unable to derive model_path from weight_path, "
            "please provide a valid Hugging Face repository id."
        )
        if len(path_pieces) >= 3:
            repo_id = f"{path_pieces[0]}/{path_pieces[1]}"
            file_name = "/".join(path_pieces[2:])

            if model_path != "" and repo_id == model_path:
                return Path(file_name), None
            elif huggingface_hub.file_exists(repo_id, file_name):
                return Path(file_name), repo_id
            elif model_path == "":
                raise ValueError(error_message)
        elif model_path == "":
            raise ValueError(error_message)

        return path, None


    @staticmethod
    def parse_diffusers_component(
        model_path: str,
        component_name: str,
        subfolder: str | None = None,
    ) -> tuple[list[Path], str | None]:
        """Parse weight paths for a specific diffusers pipeline component.

        Diffusers pipelines organize weights in subfolders (e.g., vae/, transformer/).
        This method handles locating and returning weight paths for a single component.

        Args:
            model_path: Root model path - can be a HuggingFace repo ID or local directory.
            component_name: Component name from model_index.json (e.g., "vae", "transformer").
            subfolder: Optional explicit subfolder override. Defaults to component_name.

        Returns:
            A tuple of (weight_paths, weights_repo_id):
            - weight_paths: List of Path objects pointing to weight files
            - weights_repo_id: HuggingFace repo ID if weights are remote, else None

        Examples:
            >>> WeightPathParser.parse_diffusers_component(
            ...     "black-forest-labs/FLUX.1-dev", "vae"
            ... )
            ([Path("diffusion_pytorch_model.safetensors")], "black-forest-labs/FLUX.1-dev")

            >>> WeightPathParser.parse_diffusers_component(
            ...     "/local/path/to/model", "transformer"
            ... )
            ([Path("/local/path/to/model/transformer/diffusion_pytorch_model.safetensors")], None)
        """
        component_subfolder = subfolder or component_name
        model_path_obj = Path(model_path)

        # Check if it's a local directory
        if model_path_obj.exists() and model_path_obj.is_dir():
            component_dir = model_path_obj / component_subfolder

            if not component_dir.exists():
                logger.warning(
                    f"Component subfolder not found: {component_dir}"
                )
                return [], None

            # Collect safetensor files (preferred) or pytorch files
            safetensor_files = sorted(component_dir.glob("*.safetensors"))
            if safetensor_files:
                return safetensor_files, None

            pytorch_files = sorted(component_dir.glob("*.bin"))
            if pytorch_files:
                return pytorch_files, None

            logger.warning(f"No weight files found in {component_dir}")
            return [], None

        # Assume it's a HuggingFace repo ID
        # Return relative paths - actual downloading happens elsewhere
        weight_files = []
        hf_repo_id = model_path

        # Common diffusers weight file patterns
        common_patterns = [
            f"{component_subfolder}/diffusion_pytorch_model.safetensors",
            f"{component_subfolder}/model.safetensors",
            f"{component_subfolder}/pytorch_model.bin",
        ]

        # For sharded weights, we'd need to check model.safetensors.index.json
        # For now, return the common single-file pattern
        # The actual resolution happens at load time via huggingface_hub

        for pattern in common_patterns:
            # Use the first pattern as the expected file
            weight_files.append(Path(pattern))
            break

        return weight_files, hf_repo_id

    @staticmethod
    def is_diffusers_repo(model_path: str) -> bool:
        """Check if a model path points to a diffusers-style repository.

        Args:
            model_path: Path to check (local directory or HF repo ID).

        Returns:
            True if the path contains a model_index.json file.
        """
        model_path_obj = Path(model_path)

        # Local directory check
        if model_path_obj.exists() and model_path_obj.is_dir():
            return (model_path_obj / "model_index.json").exists()

        # For HF repos, we'd need to make a network call
        # Return False for now - actual check should use huggingface_hub.file_exists
        if "/" in model_path:
            try:
                return huggingface_hub.file_exists(model_path, "model_index.json")
            except Exception:
                return False

        return False

