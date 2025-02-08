# uavsar-lidar-ml-project2


[![image](https://img.shields.io/pypi/v/uavsar-lidar-ml-project2.svg)](https://pypi.python.org/pypi/uavsar-lidar-ml-project2)
[![image](https://img.shields.io/conda/vn/conda-forge/uavsar-lidar-ml-project2.svg)](https://anaconda.org/conda-forge/uavsar-lidar-ml-project2)


**ML-based InSAR models for snow depth estimation in the central mountains of ID.**


-   Free software: MIT License
-   Documentation: https://Ibrahim-Ola.github.io/uavsar-lidar-ml-project2
    

## Features

-   TODO


# 📌 InSAR ML

## 📖 Overview
A machine learning for snow density estimation project.

This repository includes:
- Source code for **uavsar-lidar-ml-project2**.
- A reproducible Conda environment.
- Instructions for setup and usage.

---

## ⚙️ Installation and Setup

### 1️⃣ Install Conda
If you don’t have Conda installed, download **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** or **[Anaconda](https://www.anaconda.com/)**.

### 2️⃣ Clone This Repository
```bash
git clone git@github.com:cryogars/density-models.git
cd your-repo-name
```

### 3️⃣ Create and Activate the Conda Environment
Run the following commands to create a reproducible Conda environment:
```bash
conda env create --file environment.yml
conda activate my_project_env  # Use the name defined in environment.yml
```

### 4️⃣ Verify Installation
Ensure everything is set up correctly:
```bash
python --version  # Should match the version in environment.yml
conda list  # Displays installed packages
```

### 5️⃣ Updating the Environment
If you install a new package, manually add it to `environment.yml`, then update the environment:
```bash
conda env update --file environment.yml --prune
```

### 6️⃣ Deactivating and Removing the Environment
To deactivate the environment:
```bash
conda deactivate
```
To completely remove the environment:
```bash
conda env remove --name my_project_env
```

---

## 🚀 Usage
**(Explain how users should use your project. Provide examples, command-line instructions, or API usage if applicable.)**

```bash
python main.py  # Example of running the project
```

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.

---

## 📧 Contact
For any questions or issues, please open an **issue** or reach out to **ibrahimolalekana@u.boisestate.edu**.

---

🚀 Happy coding! 🎉

