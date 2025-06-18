# Letterboxd Recommender

A self-contained movie recommendation system that provides personalized suggestions based on your Letterboxd viewing history or manual movie input. The system uses a Sequential Recommendation model (SASRec) trained on the MovieLens 32M dataset to generate accurate recommendations.

## 🎬 Features

- **Letterboxd Integration**: Automatically fetch your recent films from any public Letterboxd profile
- **Manual Input**: Enter movie titles directly for recommendations
- **Rating-Based Filtering**: Get recommendations based on your highly-rated films (3★+, 4★+, 5★ only)
- **Real-time Processing**: Client-side inference using ONNX.js for fast, private recommendations
- **Self-Contained**: No external API dependencies - everything runs locally

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LetterboxdReccomender.git
   cd LetterboxdReccomender
   ```

2. **Start the local server**
   ```bash
   python serve.py
   ```

3. **Open your browser**
   Navigate to `http://localhost:8000` and start getting recommendations!

## 🏗️ Architecture Overview

### Frontend (Web Interface)
- **Pure JavaScript**: No frameworks, lightweight and fast
- **ONNX.js Runtime**: Client-side model inference
- **CORS Proxy**: Fetches Letterboxd data without backend dependencies
- **Responsive Design**: Works on desktop and mobile

### Backend (Training & Data Processing)
- **PyTorch**: Model training and validation
- **SASRec Algorithm**: Sequential recommendation using self-attention
- **MovieLens 32M**: 32 million ratings across 87,000+ movies
- **ONNX Export**: Model converted for web deployment

## 🧠 Algorithm Design

### SASRec (Self-Attentive Sequential Recommendation)

The recommendation system is built on the SASRec architecture, which excels at understanding sequential patterns in user behavior.

**Key Components:**
- **Self-Attention Mechanism**: Captures relationships between movies in your viewing history
- **Positional Encoding**: Understands the order of movies you've watched
- **Multi-Head Attention**: Focuses on different aspects of movie preferences simultaneously

**Training Process:**
1. **Data Preprocessing** ([`code/main.py`](code/main.py))
   - Filters users with ≥10 movie ratings
   - Calculates 70th percentile ratings per user
   - Creates binary positive feedback (above percentile = liked)

2. **Sequence Generation** ([`code/train.py`](code/train.py))
   - Converts user ratings into sequential movie-watching patterns
   - Pads sequences to consistent length (50 movies max)
   - Splits into training/validation sets (80/20)

3. **Model Training**
   - **Architecture**: 2 transformer layers, 8 attention heads, 128 embedding dimensions
   - **Loss Function**: Binary cross-entropy with negative sampling
   - **Optimization**: Adam optimizer with early stopping
   - **Validation**: Tracks accuracy and prevents overfitting

4. **ONNX Conversion** ([`code/convert.py`](code/convert.py))
   - Exports trained PyTorch model to ONNX format
   - Optimizes for web deployment and inference speed

### Model Performance
- **Training Data**: 200K+ users, 32M+ ratings
- **Vocabulary**: 87K+ unique movies
- **Context Length**: Up to 50 previous movies
- **Inference Time**: <100ms client-side

## 📁 Project Structure

```
LetterboxdReccomender/
├── web/                    # Frontend application
│   ├── index.html         # Main web interface
│   ├── app.js            # Core application logic
│   ├── letterboxd.js     # Letterboxd integration
│   └── style.css         # UI styling
├── code/                  # Training and processing scripts
│   ├── main.py           # Data preprocessing pipeline
│   ├── train.py          # Model training with validation
│   ├── convert.py        # ONNX model conversion
│   ├── inference.py      # Python inference testing
│   ├── mapping.py        # Movie ID/title mappings
│   └── depickle.py       # Model inspection utilities
├── data/                  # MovieLens dataset
│   ├── movies.csv        # Movie metadata
│   ├── ratings.csv       # User ratings (32M+ entries)
│   └── README.txt        # Dataset documentation
├── models/                # Trained models and mappings
│   ├── sasrec_model.onnx # Web-optimized model
│   ├── metadata.json     # Model configuration
│   ├── movie_map.pkl     # Movie ID mappings
│   └── *.pth            # PyTorch checkpoints
└── output/               # Processing outputs and logs
```

## 🔧 Technical Implementation

### Data Flow

1. **Input Processing**
   - Letterboxd: Scrapes user profile via CORS proxy
   - Manual: Parses movie titles with fuzzy matching
   - Normalizes titles and maps to MovieLens IDs

2. **Sequence Preparation**
   - Left-pads movie sequence to 50 items
   - Converts to appropriate tensor format
   - Handles various input lengths gracefully

3. **Model Inference**
   - ONNX.js runs model in browser
   - Processes attention weights across sequence
   - Generates probability scores for all movies

4. **Recommendation Generation**
   - Excludes already-watched movies
   - Ranks by confidence score
   - Returns top 10 suggestions with metadata

### Key Files Explained

**[`web/app.js`](web/app.js)**: Core frontend logic
- Model loading and initialization
- Movie title normalization and matching
- ONNX inference pipeline
- UI state management

**[`web/letterboxd.js`](web/letterboxd.js)**: Letterboxd integration
- Profile scraping with CORS proxy
- Rating extraction and processing
- Film matching against MovieLens database

**[`code/train.py`](code/train.py)**: Model training pipeline
- SASRec implementation in PyTorch
- Training loop with validation
- Model checkpointing and saving

**[`code/main.py`](code/main.py)**: Data preprocessing
- Rating threshold calculation (70th percentile)
- User/movie filtering
- Binary feedback generation

## 🎯 Self-Contained Design

This project is designed to run completely independently:

**No External APIs Required:**
- Letterboxd data fetched via public CORS proxy
- All inference happens client-side
- MovieLens data included in repository

**Offline Capability:**
- Once loaded, works without internet
- All models and data bundled locally
- No tracking or data collection

**Easy Deployment:**
- Single Python file serves everything
- No database or complex setup needed
- Works on any system with Python 3.6+

## 🚀 Getting Started (Detailed)

### Prerequisites
- Python 3.6+ with basic libraries (pandas, numpy)
- Modern web browser with JavaScript enabled
- ~500MB disk space for models and data

### Installation

1. **Download the project**
   ```bash
   git clone https://github.com/yourusername/LetterboxdReccomender.git
   cd LetterboxdReccomender
   ```

2. **Install Python dependencies** (if training models)
   ```bash
   pip install torch pandas numpy scikit-learn onnx
   ```

3. **Start the web server**
   ```bash
   python serve.py
   ```

4. **Access the application**
   Open `http://localhost:8000` in your browser

### Usage

**Option 1: Letterboxd Integration**
1. Enter any public Letterboxd username
2. System fetches recent rated films automatically
3. Choose recommendation source:
   - Recent films (latest 50)
   - 5-star films only
   - 4+ star films
   - 3+ star films

**Option 2: Manual Input**
1. Enter comma-separated movie titles
2. System matches against MovieLens database
3. Get recommendations based on your list

## 🔬 Model Training (Advanced)

If you want to retrain the model or experiment with different parameters:

### 1. Data Preparation
```bash
cd code
python main.py  # Processes MovieLens data
```

### 2. Model Training
```bash
python train.py  # Trains SASRec model with validation
```

### 3. ONNX Conversion
```bash
python convert.py  # Converts to web-compatible format
```

### Training Configuration
Edit [`code/train.py`](code/train.py) to modify:
- `EMBED_DIM`: Embedding dimensions (default: 128)
- `NUM_HEADS`: Attention heads (default: 8)
- `NUM_LAYERS`: Transformer layers (default: 2)
- `MAX_LEN`: Sequence length (default: 50)
- `BATCH_SIZE`: Training batch size (default: 1024)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Dataset Attribution

This project uses the MovieLens 32M dataset:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GroupLens Research** for the MovieLens dataset
- **Microsoft** for ONNX.js runtime
- **Letterboxd** for the inspiration and public profiles
- **SASRec authors** for the sequential recommendation algorithm

## 🐛 Troubleshooting

**Model won't load:**
- Check browser console for errors
- Ensure all files in `models/` directory exist
- Try a different browser (Chrome/Firefox recommended)

**Letterboxd fetch fails:**
- Verify the username is correct and profile is public
- Check CORS proxy status
- Try manual input as alternative

**No recommendations generated:**
- Ensure matched movies > 0
- Check that movie titles match MovieLens database
- Try different/more popular movie titles

## 🔮 Future Improvements

- [ ] Support for more rating platforms (IMDb, TMDb)
- [ ] Collaborative filtering integration
- [ ] Genre-based filtering options
- [ ] Movie poster and metadata display
- [ ] Recommendation explanations
- [ ] Export/save recommendation lists