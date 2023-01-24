# Treebeard_ViT
## Install & Run
### 1. set configs
* plant-classification_template/config/vit16small224/config.yaml
* plant-classification_template/config/vit16small224/accelerator.yaml
* docker-compose.yaml (environment variable, network option, etc.)
### 2. docker compose run
``` shell
git clone https://github.com/stonehye/plant-classification_template.git && cd plant-classification_template
docker compose up --build # build & run
docker compose down # remove docker container, image, network
```
## Related docs
* [Treebeard project notion page](https://www.notion.so/arbeon/Treebeard-04a2a205e28c4d259086a34c84375173)
