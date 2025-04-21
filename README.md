# AI-powered Schema Matching

An intelligent system that uses machine learning to automatically match and standardize database column names across different schemas.

## Overview

This project implements an AI-powered solution for schema matching, which helps in standardizing column names across different databases or data sources. It uses XGBoost for classification and TF-IDF vectorization to understand the semantic meaning of column names.

## Features

- Automated column name standardization
- REST API interface for easy integration
- Web-based user interface
- Machine learning-powered matching using XGBoost
- TF-IDF vectorization for semantic understanding

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-powered-Schema-matching.git
cd AI-powered-Schema-matching
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. To use the API endpoint:
   - Send a POST request to `/predict_api`
   - Include column names in the request body as JSON
   - Example:
     ```json
     {
         "data": ["customer_id", "first_name", "last_name", "email_address"]
     }
     ```

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates for the web interface
- `xgboost_schema_mapper.pkl` - Trained XGBoost model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text preprocessing
- `label_encoder.pkl` - Label encoder for mapping predictions

## Dependencies

- Flask - Web framework
- scikit-learn - Machine learning utilities
- XGBoost - Gradient boosting framework
- pandas - Data manipulation
- joblib - Model persistence
- matplotlib & seaborn - Data visualization

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.