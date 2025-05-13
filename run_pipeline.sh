echo "Setting up the environment..."
pip install -r requirements.txt

echo "Installing Spacy and NLTK data..."
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -u src/scripts/nltk_setup.py


echo "Running the pipeline..."

echo "Running the translation service if necessary..."
#python -u services/translate.py


echo "Running the data service..."
python -u services/data.py


echo "Running the complexity metrics analysis service..."
python -u services/complexity_metrics.py

echo "Running the inference service..."
python -u services/infer.py
