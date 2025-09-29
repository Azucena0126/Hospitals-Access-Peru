## Hospital Access Perú with Python

### Running locally

To run locally, it is advisable to first create a virtual environment

You will need to have Python installed and in your `PATH`. While located in the root directory of this repository, 

```
python -m venv env
```

Now activate the virtual environment:

#### Windows

```
env/Scripts/activate
```

Now you must install the necessary dependencies to run the dashboard:

```
pip install -r requirements.txt
```

This should take a few minutes. Once it is done, you can run the streamlit application:

```
streamlit run src/streamlit_app.py
```

This should start a locally hosted server and automatically open a browser tab with the application
