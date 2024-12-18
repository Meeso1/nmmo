import os
import pickle
import time
import torch
import numpy as np
from typing import Any, ClassVar, Literal


class Jar:
    """A pickle-based object storage class."""

    base_path: ClassVar[str] = ".jar"

    def __init__(self, prefix: str = "") -> None:
        """
        Initialize the Jar instance with a path prefix.

        Args:
            prefix (str): Path prefix for all objects saved/retrieved using this instance.
        """
        self.prefix: str = prefix

    def _get_dir_path(self, name: str) -> str:
        name_dir = os.path.dirname(name)
        return os.path.join(self.base_path, self.prefix, name_dir)

    def _get_full_path(self, name: str, kind: Literal["numpy", "torch", "pickle"]) -> str:
        dir_path = self._get_dir_path(name)
        timestamp = int(time.time())
        extension = ".pkl" if kind == "pickle" else (".npy" if kind == "numpy" else ".pth")

        filename = f"{os.path.basename(name)}-{timestamp}{extension}"
        return os.path.join(dir_path, filename)

    def _find_files(self, name: str) -> list[str]:
        dir_path = self._get_dir_path(name)
        if not os.path.exists(dir_path):
            return []
        files = [f for f in os.listdir(dir_path)
                 if f.startswith(os.path.basename(name) + '-')
                 and (f.endswith('.pkl') or f.endswith('.npy') or f.endswith('.pth'))]
        return [os.path.join(dir_path, f) for f in files]

    def _get_latest_file(self, name: str) -> str:
        files = self._find_files(name)
        if not files:
            raise KeyError(f"No object named '{name}' found")

        timestamps = {
            f: int(f.split('-')[-1].split('.')[0])
            for f in files
        }
        latest_file = max(timestamps, key=timestamps.get)
        return latest_file

    def add(self, name: str, obj: Any, kind: Literal["numpy", "torch", "pickle"] = "pickle") -> None:
        """
        Save the object to a file with a timestamp.

        Args:
            name (str): The name of the object.
            obj (Any): The object to save.
        """
        full_path = self._get_full_path(name, kind)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if kind == "numpy":
            np.save(full_path, obj)
        elif kind == "torch":
            torch.save(obj, full_path)
        else:
            with open(full_path, 'wb') as f:
                pickle.dump(obj, f)

    def get(self, name: str) -> Any:
        """
        Load the most recent object with the given name.

        Args:
            name (str): The name of the object.

        Returns:
            Any: The loaded object.
        """
        full_path = self._get_latest_file(name)
        if full_path.endswith('.npy'):
            return np.load(full_path)
        elif full_path.endswith('.pth'):
            return torch.load(full_path)
        else:
            with open(full_path, 'rb') as f:
                return pickle.load(f)

    def get_all(self, name: str) -> list[Any]:
        """
        Load all objects with the given name.

        Args:
            name (str): The name of the object.

        Returns:
            list[Any]: List of loaded objects.
        """
        files = self._find_files(name)
        return [self.get(f) for f in files]

    def remove_latest(self, name: str) -> None:
        """
        Delete the most recent object with the given name.

        Args:
            name (str): The name of the object.
        """
        full_path = self._get_latest_file(name)
        os.remove(full_path)

    def remove_all(self, name: str) -> None:
        """
        Delete all objects with the given name.

        Args:
            name (str): The name of the object.
        """
        files = self._find_files(name)
        for f in files:
            os.remove(f)

    def __contains__(self, name: str) -> bool:
        """
        Check if any object with the given name exists.

        Args:
            name (str): The name of the object.

        Returns:
            bool: True if the object exists, False otherwise.
        """
        return len(self._find_files(name)) > 0

    def object_names(self) -> set[str]:
        """
        Return names of all objects contained in the Jar.

        Returns:
            set[str]: Set of object names.
        """
        object_names = set()
        base_dir = os.path.join(self.base_path, self.prefix)
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not (file.endswith('.pkl') or file.endswith('.npy') or file.endswith('.pth')):
                    continue

                relative_path = os.path.relpath(root, base_dir)
                name = '-'.join(file.split('-')[:-1]) # Remove part after last '-' (timestamp)
                object_name = os.path.join(relative_path, name).replace('\\', '/')
                object_names.add(object_name)
        return object_names
