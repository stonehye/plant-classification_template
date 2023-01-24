# Treebeard_ViT
## Install & Run
### 1. set configs
* treebeard_vit/config/vit16small224/config.yaml
* treebeard_vit/config/vit16small224/accelerator.yaml
* docker-compose.yaml (environment variable, network option, etc.)
### 2. docker compose run
``` shell
git clone https://gitlab.dev-merge.com/merge-development/ai-sg/ai-model/treebeard/treebeard_vit.git && cd treebeard_vit
docker compose up --build # build & run
docker compose down # remove docker container, image, network
```
## Related docs
* [Treebeard project notion page](https://www.notion.so/arbeon/Treebeard-04a2a205e28c4d259086a34c84375173)