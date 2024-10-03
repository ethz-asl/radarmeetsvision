docker buildx build -t omavteam/radarmeetsvision:blearn-latest -f blearn/Dockerfile --push .
docker buildx build -t omavteam/radarmeetsvision:latest -f desktop/Dockerfile --push .
