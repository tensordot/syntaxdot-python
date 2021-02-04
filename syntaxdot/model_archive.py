import os
from pathlib import Path
from tarfile import TarFile

import requests

from .syntaxdot import Annotator

def cache_dir():
    return Path(os.getenv("SYNTAXDOT_CACHE_DIR", Path.home() / ".cache" / "syntaxdot"))


class ModelArchive:
    """
    Class for downloading and loading annotators provided by TensorDot.
    """

    def __init__(self):
        pass

    def load(self, model_name):
        """
        Load the model `model_name`. The model will be used from the local
        cache if it was used previously. Otherwise, it will be downloaded
        to the cache.

        :param model_name: The name of the model.
        """

        model_path = self._get_from_cache_or_download(model_name)
        return Annotator(str(model_path))


    def _get_from_cache_or_download(self, model_name):
        cache_path = cache_dir()
        if not cache_path.exists():
            cache_path.mkdir(parents=True)

        model_path = cache_path / model_name

        if model_path.exists():
            return model_path / "syntaxdot.conf"

        tarball = (cache_path / model_name).with_suffix(".tar.gz")
        with requests.get(f"https://s3.tensordot.com/syntaxdot/models/{model_name}.tar.gz", stream=True) as request:
            request.raise_for_status()

            with open(tarball, "wb") as f:
                for chunk in request.iter_content(chunk_size=65536):
                    f.write(chunk)

        # Todo: add model verification.

        with TarFile.open(tarball) as tar:
            tar.extractall(cache_path)

        tarball.unlink()

        return model_path / "syntaxdot.conf"





