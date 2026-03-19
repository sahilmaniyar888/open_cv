"""Lightweight dataset versioning with manifest tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


class DatasetVersion:
    """
    Tracks dataset versions using manifest files.

    Records which images/labels are in each version, generation
    parameters, QA results, and lineage information.
    """

    def __init__(self, dataset_dir: str, registry_dir: str | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.registry_dir = Path(registry_dir) if registry_dir else self.dataset_dir / ".versions"
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        version_name: str,
        description: str = "",
        generation_params: dict | None = None,
        qa_results: dict | None = None,
        auto_labeler_version: str | None = None,
        parent_version: str | None = None,
    ) -> dict:
        """
        Create a new version snapshot of the current dataset.

        Args:
            version_name: unique name for this version (e.g. "v1.0", "synthetic_round3")
            description: human-readable description
            generation_params: parameters used to generate this data
            qa_results: output from DatasetQA
            auto_labeler_version: which model version auto-labeled
            parent_version: which version this derives from

        Returns:
            version manifest dict
        """
        image_dir = self.dataset_dir / "images"
        label_dir = self.dataset_dir / "labels"

        image_files = sorted(f.name for f in image_dir.iterdir()
                             if f.suffix.lower() in {".jpg", ".jpeg", ".png"}) if image_dir.exists() else []
        label_files = sorted(f.name for f in label_dir.iterdir()
                             if f.suffix == ".txt") if label_dir.exists() else []

        # compute content hash
        hasher = hashlib.sha256()
        for fname in image_files + label_files:
            hasher.update(fname.encode())
        content_hash = hasher.hexdigest()[:12]

        manifest = {
            "version": version_name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": content_hash,
            "num_images": len(image_files),
            "num_labels": len(label_files),
            "image_files": image_files,
            "label_files": label_files,
            "lineage": {
                "parent_version": parent_version,
                "generation_params": generation_params,
                "auto_labeler_version": auto_labeler_version,
            },
            "qa_results": qa_results,
        }

        # save manifest
        manifest_path = self.registry_dir / f"{version_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Created version '{version_name}' with {len(image_files)} images")
        return manifest

    def list_versions(self) -> list[dict]:
        """List all available versions."""
        versions = []
        for manifest_file in sorted(self.registry_dir.glob("*.json")):
            with open(manifest_file) as f:
                data = json.load(f)
            versions.append({
                "version": data["version"],
                "created_at": data["created_at"],
                "num_images": data["num_images"],
                "content_hash": data["content_hash"],
                "description": data.get("description", ""),
            })
        return versions

    def get_version(self, version_name: str) -> dict | None:
        """Load a specific version manifest."""
        path = self.registry_dir / f"{version_name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def compare_versions(self, v1_name: str, v2_name: str) -> dict:
        """Compare two dataset versions."""
        v1 = self.get_version(v1_name)
        v2 = self.get_version(v2_name)

        if not v1 or not v2:
            return {"error": "version not found"}

        v1_images = set(v1["image_files"])
        v2_images = set(v2["image_files"])

        return {
            "v1": v1_name,
            "v2": v2_name,
            "added_images": sorted(v2_images - v1_images),
            "removed_images": sorted(v1_images - v2_images),
            "common_images": len(v1_images & v2_images),
            "v1_total": len(v1_images),
            "v2_total": len(v2_images),
        }

    def get_lineage(self, version_name: str) -> list[str]:
        """Trace version lineage back to root."""
        lineage = []
        current = version_name
        visited = set()

        while current and current not in visited:
            visited.add(current)
            lineage.append(current)
            v = self.get_version(current)
            if not v:
                break
            current = v.get("lineage", {}).get("parent_version")

        return lineage
