from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ocr4all_helper_scripts',
      version='0.4.1',
      description='Different python scripts used in the OCR4all workflow.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/OCR4all/OCR4all_helper-scripts',
      author='Nico Balbach, Maximilian Nöth',
      author_email='nico.balbach@informatik.uni-wuerzburg.de, maximilian.noeth@uni-wuerzburg.de',
      packages=find_packages(),
      license="MIT License",
      entry_points={
            'console_scripts': [
                  'ocr4all-helper-scripts = ocr4all_helper_scripts.cli: cli'
            ]
      },
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      install_requires=[
            "click",
            "numpy",
            "lxml",
            "scikit-image",
            "Pillow"
      ],
      keywords=["OCR", "optical character recognition"],
      zip_safe=False
      )
