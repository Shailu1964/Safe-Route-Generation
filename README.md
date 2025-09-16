# Crime Prediction & Safe Navigation System

## Description
This project is a web-based application designed to help users find the safest routes within Pune city by predicting crime severity and visualizing crime-prone areas. It combines machine learning, real crime datasets, and interactive maps to guide users away from high-risk zones. The system is built with Python and Flask, using data science and mapping libraries for backend processing and visualization.

### How It Works
1. **Data Collection & Preprocessing:**
   - Real crime data for Pune is collected and preprocessed for analysis.
   - Data is cleaned and transformed for use in machine learning models and route calculations.
2. **Model Training:**
   - A machine learning model is trained to predict the severity of crime in different areas based on historical data.
   - The trained model is saved as `crime_severity_model.pkl`.
3. **Route Generation:**
   - The city map is represented as a graph, with nodes as locations and edges as possible routes.
   - Each route is weighted by both distance and predicted crime severity.
   - The safest route is calculated using graph algorithms, prioritizing low-crime paths.
4. **Web Application:**
   - Users enter their source and destination.
   - The app predicts the safest route and displays it on an interactive map, highlighting crime-prone areas.
   - Users can view crime statistics and details for their selected route.

## Why These Technologies?
- **Python:** Easy to use for data science, machine learning, and web development.
- **Flask:** Lightweight web framework, ideal for small to medium applications.
- **scikit-learn, pandas, numpy:** For data processing and machine learning.
- **folium, matplotlib:** For map and data visualization.
- **Jinja2 templates:** For dynamic HTML rendering.
- **pickle:** For saving/loading trained models and data objects.

## Project Structure
```
Crime_Prediction/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── crime_severity_model.pkl# Trained ML model
├── pune_graph.pkl          # Serialized city graph
├── *.csv                   # Crime datasets
├── generate_graph.py       # Script to build the city graph
├── process.py              # Data preprocessing script
├── route_calculations.py   # Route finding logic
├── train.py                # Model training script
├── utils.py                # Helper functions
├── static/                 # Images, map outputs
├── templates/              # HTML templates
├── cache/                  # Local cache files
├── .env                    # Environment variables (if needed)
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

## Installation & Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/Shailu1964/Crime_Prediction.git
   cd Crime_Prediction/Crime_Prediction
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv venv311
   venv311\Scripts\activate  # On Windows
   # or
   source venv311/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **(Optional) Set up environment variables:**
   - Copy `.env.example` to `.env` and update as needed.
5. **Run the application:**
   ```sh
   python app.py
   ```
6. **Open your browser:**
   - Go to `http://localhost:5000`

## Usage
- Enter your source and destination in the web interface.
- The app will display the safest route and highlight crime-prone areas on the map.
- View detailed crime statistics for your route.

## Screenshots
- ![Home Page](static/images/home_placeholder.png)
- ![Map View](static/images/map_placeholder.png)

## API/Configuration Details
- **Model & Data Files:**
  - `crime_severity_model.pkl`: Machine learning model for crime prediction.
  - `pune_graph.pkl`: Graph object for route calculations.
  - CSV files: Contain raw and processed crime data.
- **Configuration:**
  - Environment variables (API keys, Flask settings) can be set in `.env`.

## Troubleshooting
- If you encounter missing package errors, ensure your virtual environment is activated and all dependencies are installed.
- For issues with map rendering, check that the `static/map/` directory is writable.
- Logs are saved in `app.log` for debugging.

## Contributing
1. Fork this repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -am 'Add new feature'`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request.

## Contact / Author
- **Author:** Shailu1964
- **GitHub:** [Shailu1964](https://github.com/Shailu1964)
- **Email:** yerawad.shailesh@gmail.com

---
If you have questions or suggestions, feel free to open an issue or contact the author.
