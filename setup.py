from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="syntaxdot",
    version="0.3.0-beta.0",
    rust_extensions=[RustExtension("syntaxdot.syntaxdot", binding=Binding.PyO3)],
    packages=["syntaxdot"],
    zip_safe=False,
)