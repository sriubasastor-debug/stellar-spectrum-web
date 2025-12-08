pyinstaller \
 --onefile \
 --add-data "templates:templates" \
 --add-data "static:static" \
 app.py

echo "Build complete!"
