#!/bin/env bash

echo "Checking for resource data..."
if [ ! -f resources.tar.gz ]; then
  curl -o resources.tar.gz https://cmi4earth.blob.core.windows.net/cmi-antbok/resources.tar.gz
fi

echo "Extracting resource data..."
if [ ! -d resources ]; then
  tar -zxf resources.tar.gz
fi

echo "Done."
