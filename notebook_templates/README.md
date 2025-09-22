# Notebook Templates

This directory contains clean Python templates that can be used as references for creating Jupyter notebooks.

## Important Security Notes

⚠️ **NEVER hardcode credentials in notebooks!**

Instead:
1. Use environment variables (see `.env.example`)
2. Load credentials using `python-dotenv`
3. Keep notebooks in the `notebooks/` directory (which is gitignored)

## Available Templates

### data_fetch_template.py
Shows how to safely connect to a database using environment variables.

## Converting to Notebooks

To use these templates in Jupyter:
1. Copy the template to the `notebooks/` directory
2. Rename with `.ipynb` extension
3. Open in Jupyter and convert cells as needed
4. Remember: notebooks directory is gitignored for security

## Example .env Setup

```bash
DB_SERVER=your_server
DB_DATABASE=your_database
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

Never commit the actual `.env` file!