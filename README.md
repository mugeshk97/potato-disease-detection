# potato-disease-detection

command to run the tf-serving 

```
tensorflow_model_server --rest_api_port=8601  --allow_version_labels_for_unavailable_models --model_config_file=model.config
```

nohup cmds to run in background

```
nohup python app.py &> fastapi.logs &
```

same cmd to run the tf model server in background