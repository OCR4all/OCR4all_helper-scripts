from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='ocr4all_helpers',
      version='0.3.1',
      description='Different python scripts used in the OCR4all workflow.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/OCR4all/OCR4all_helper-scripts',
      author='Nico Balbach, Maximilian Nöth',
      author_email='nico.balbach@informatik.uni-wuerzburg.de, maximilian.noeth@protonmail.com',
      packages=find_packages(),
      license='MIT License',
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent",
          ],
      entry_points={
            'console_scripts': [
                  'pagelineseg=ocr4all_helpers.pagelineseg:cli',
                  'skewestimate=ocr4all_helpers.skewestimate:cli',
                  'pagedir2pagexml=ocr4all_helpers.pagedir2pagexml:main',
                  'legacy_convert=ocr4all_helpers.legacyconvert:main'
            ],
      },
      install_requires=open("requirements.txt").read().split(),
      python_requires='>=3.6',
      zip_safe=False
      )
