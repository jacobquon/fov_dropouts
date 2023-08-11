from setuptools import setup
import pathlib

HERE = pathlib.Path(__file__).parent

REQUIREMENTS = (HERE / "requirements.txt").read_text()
install_requirements = REQUIREMENTS.splitlines()

setup(
	name='dropout_detection',
	verion='1.0.0',
	scripts=['dropout_detection.py'],
	install_requires=install_requirements,
	py_modules=[]
)
