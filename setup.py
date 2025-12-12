"""Setup script for backward compatibility with pip install."""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eyemouse",
    version="0.1.0",
    author="Eye Control Mouse Contributors",
    description="Hands-free mouse control using webcam and eye tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sohaibbasharat1384/eye-control-mouse",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=requirements,
    python_requires=">=3.9,<3.12",
    entry_points={
        "console_scripts": [
            "eyemouse=eyemouse.app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
    keywords="eye-tracking gaze accessibility mouse-control computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/sohaibbasharat1384/eye-control-mouse/issues",
        "Source": "https://github.com/sohaibbasharat1384/eye-control-mouse",
        "Documentation": "https://github.com/sohaibbasharat1384/eye-control-mouse",
    },
)
