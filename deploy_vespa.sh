#!/bin/bash

echo "Creating Vespa application package..."

# Create temporary directory for the app package
mkdir -p temp_vespa_app

# Copy configuration files
cp -r config/vespa/* temp_vespa_app/

rm -rf ../config/vespa-app.zip

# Create application package zip
cd temp_vespa_app
zip -r ../config/vespa-app.zip *
cd ..

# Clean up
rm -rf temp_vespa_app

echo "Deploying to Vespa..."

# Deploy the application
curl -X POST -H "Content-Type: application/zip" \
  --data-binary @config/vespa-app.zip \
  http://localhost:19071/application/v2/tenant/default/prepareandactivate

echo "Deployment complete!"
echo "Waiting for Vespa to be ready..."

# Wait for Vespa to be ready
for i in {1..30}; do
  if curl -s http://localhost:4080/state/v1/health | grep -q '"code" : "up"'; then
    echo "Vespa is ready!"
    break
  fi
  echo "Waiting... ($i/30)"
  sleep 2
done

# Check final status
curl http://localhost:4080/state/v1/health | python -m json.tool