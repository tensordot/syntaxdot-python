from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="syntaxdot",
    version="0.3.0b1",
    rust_extensions=[RustExtension("syntaxdot.syntaxdot", binding=Binding.PyO3)],
    packages=["syntaxdot"],
    install_requires=["requests"],
    zip_safe=False,
)