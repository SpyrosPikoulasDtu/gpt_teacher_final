# Setup

## 1. Clone the Repository
Clone the `gpt_teacher_final` repository to your local machine using the GitHub CLI or command-line:

```bash
git clone https://github.com/SpyrosPikoulasDtu/gpt_teacher_final.git
```

Navigate into the cloned repository.

```bash
cd gpt_teacher_final
```

## 2. Set up the environment

Create a new virtual environment.

```bash
python -m venv venv
```

Activate the new environment.

```bash
.\venv\Scripts\activate
```

## 3. Install the requirements

```bash
pip install -r requirements.txt
```

## 4. Download the app data

Download the `data` folder from here: [Data folder](https://drive.google.com/drive/folders/1BQoijinmjCjfA7Bd1o4cYmn8r1RX_jCe?usp=sharing)

Place the folder(not the files) in the same folder as the `app.py`(the gpt_teacher_final folder)

## 5. Run the app

```bash
python -m streamlit run app.py
```
