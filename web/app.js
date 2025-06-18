let session = null;
let metadata = null;
let movieIdToName = null;
let movieTitles = {}; // For normalized title to ID mapping (like Python)
let letterboxdClient = null;

// Tab switching functionality
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName + 'Tab').classList.add('active');
    
    // Add active class to selected tab
    event.target.classList.add('active');
}

// Wait for ONNX Runtime to load
function waitForOrt() {
    return new Promise((resolve, reject) => {
        if (typeof ort !== 'undefined') {
            resolve();
            return;
        }
        
        let attempts = 0;
        const maxAttempts = 50; // 5 seconds max
        
        const checkOrt = () => {
            attempts++;
            if (typeof ort !== 'undefined') {
                resolve();
            } else if (attempts >= maxAttempts) {
                reject(new Error('ONNX Runtime failed to load'));
            } else {
                setTimeout(checkOrt, 100);
            }
        };
        
        checkOrt();
    });
}

// Initialize the application
async function init() {
    try {
        updateStatus('Loading ONNX Runtime...', 'loading');
        
        // Wait for ONNX Runtime to be available
        await waitForOrt();
        
        updateStatus('Loading model metadata...', 'loading');
        
        // Load metadata
        const metadataResponse = await fetch('models/metadata.json');
        if (!metadataResponse.ok) {
            throw new Error(`Failed to load metadata: ${metadataResponse.status} ${metadataResponse.statusText}`);
        }
        metadata = await metadataResponse.json();
        
        updateStatus('Loading model names...', 'loading');
        
        // Load movie ID to name mapping
        const movieNamesResponse = await fetch('models/movie_id_to_name.json');
        if (!movieNamesResponse.ok) {
            throw new Error(`Failed to load movie names: ${movieNamesResponse.status} ${movieNamesResponse.statusText}`);
        }
        const movieNamesData = await movieNamesResponse.json();
        movieIdToName = movieNamesData.data;
        
        // Build normalized title to ID mapping (like Python's movie_titles)
        buildMovieTitlesMapping();
        
        // Initialize Letterboxd client
        letterboxdClient = new LetterboxdClient();
        letterboxdClient.setMovieData(movieIdToName, movieTitles);
        
        updateStatus('Loading ONNX model...', 'loading');
        
        // Load ONNX model with error handling
        try {
            session = await ort.InferenceSession.create('models/sasrec_model.onnx', {
                executionProviders: ['wasm']
            });
        } catch (modelError) {
            console.error('Model loading error:', modelError);
            throw new Error(`Failed to load ONNX model: ${modelError.message}`);
        }
        
        updateStatus('Model loaded successfully!', 'success');
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('fetchLetterboxdBtn').disabled = false;
        
        // Display model info
        displayDebugInfo();
        
    } catch (error) {
        console.error('Error loading model:', error);
        updateStatus(`Error loading model: ${error.message}`, 'error');
    }
}

// Normalize title function - updated to preserve years when requested
function normalizeTitle(title, removeYear = true) {
    if (!title || typeof title !== 'string') {
        return '';
    }
    
    let normalized = title.toLowerCase().trim();
    
    // Remove year in parentheses at the end only if requested
    if (removeYear) {
        normalized = normalized.replace(/\s*\(\d{4}\)\s*$/, '');
    }
    
    return normalized;
}

function buildMovieTitlesMapping() {
    movieTitles = {};
    if (!movieIdToName) return;
    
    for (const [movieId, title] of Object.entries(movieIdToName)) {
        if (title) {
            // Store both versions: with year and without year
            const withYear = normalizeTitle(title, false); // Keep year
            const withoutYear = normalizeTitle(title, true); // Remove year
            
            if (withYear && withYear.trim() !== '') {
                movieTitles[withYear] = parseInt(movieId);
            }
            if (withoutYear && withoutYear.trim() !== '' && withoutYear !== withYear) {
                // Only add without year if it's different from with year
                movieTitles[withoutYear] = parseInt(movieId);
            }
        }
    }
}

// Update status message
function updateStatus(message, type) {
    const statusDiv = document.getElementById('modelStatus');
    statusDiv.textContent = message;
    statusDiv.className = type;
}

// Update Letterboxd status message
function updateLetterboxdStatus(message, type) {
    const statusDiv = document.getElementById('letterboxdStatus');
    statusDiv.innerHTML = `<div class="${type}">${message}</div>`;
}

// Display debug information
function displayDebugInfo() {
    const debugDiv = document.getElementById('debugInfo');
    const movieCount = movieIdToName ? Object.keys(movieIdToName).length : 0;
    debugDiv.innerHTML = `
        <p><strong>Model Info:</strong></p>
        <ul>
            <li>Max sequence length: ${metadata.model_params.max_len}</li>
            <li>Number of movies: ${metadata.num_movies}</li>
            <li>Embedding dimension: ${metadata.model_params.embed_dim}</li>
            <li>Input shape: [${metadata.input_shape.join(', ')}]</li>
            <li>Output shape: [${metadata.output_shape.join(', ')}]</li>
            <li>Movie names loaded: ${movieCount} movies</li>
            <li>Normalized titles: ${Object.keys(movieTitles).length} movies</li>
            <li>ONNX Runtime loaded: ${typeof ort !== 'undefined' ? 'Yes' : 'No'}</li>
        </ul>
    `;
}

// Get movie name by ID
function getMovieName(movieId) {
    return movieIdToName && movieIdToName[movieId.toString()] 
        ? movieIdToName[movieId.toString()] 
        : `Unknown Movie (ID: ${movieId})`;
}

// Clear Letterboxd data and reset the interface
function clearLetterboxdData() {
    // Reset letterboxd client data
    if (letterboxdClient) {
        letterboxdClient.matchedIds = [];
        letterboxdClient.allMatchedFilms = [];
    }
    
    // Clear input field
    document.getElementById('letterboxdUsername').value = '';
    
    // Hide and clear all sections
    document.getElementById('letterboxdOptions').style.display = 'none';
    document.getElementById('letterboxdFilms').style.display = 'none';
    document.getElementById('letterboxdFilms').innerHTML = '';
    document.getElementById('letterboxdRecommendations').innerHTML = '';
    document.getElementById('letterboxdStatus').innerHTML = '';
    document.getElementById('ratingButtons').innerHTML = '';
    
    // Hide clear button
    document.getElementById('clearLetterboxdBtn').style.display = 'none';
    
    // Re-enable fetch button if it was disabled
    document.getElementById('fetchLetterboxdBtn').disabled = false;
    document.getElementById('fetchLetterboxdBtn').textContent = 'Fetch Films';
}

// Fetch films from Letterboxd
async function fetchLetterboxdFilms() {
    const username = document.getElementById('letterboxdUsername').value.trim();
    
    if (!username) {
        alert('Please enter a Letterboxd username');
        return;
    }
    
    try {
        document.getElementById('fetchLetterboxdBtn').disabled = true;
        document.getElementById('fetchLetterboxdBtn').textContent = 'Fetching...';
        updateLetterboxdStatus('Fetching films from Letterboxd...', 'loading');
        
        const result = await letterboxdClient.processUserFilms(username);
        
        // Display results
        displayLetterboxdFilms(result.matchedFilms, result.unmatchedFilms);
        
        if (result.matchedFilms.length > 0) {
            updateLetterboxdStatus(`Found ${result.matchedFilms.length} recent films`, 'success');
            
            // Show recommendation options
            displayRecommendationOptions(result.availableRatings);
            
            document.getElementById('letterboxdOptions').style.display = 'block';
            // Show clear button after successful fetch
            document.getElementById('clearLetterboxdBtn').style.display = 'inline-block';
        } else {
            updateLetterboxdStatus('No matching films found in our database', 'error');
        }
        
    } catch (error) {
        console.error('Error fetching Letterboxd films:', error);
        updateLetterboxdStatus(`Error: ${error.message}`, 'error');
    } finally {
        document.getElementById('fetchLetterboxdBtn').disabled = false;
        document.getElementById('fetchLetterboxdBtn').textContent = 'Fetch Films';
    }
}

// Display recommendation options based on available ratings
function displayRecommendationOptions(availableRatings) {
    const ratingButtonsDiv = document.getElementById('ratingButtons');
    
    if (availableRatings.length === 0) {
        ratingButtonsDiv.innerHTML = '<p class="no-ratings">No rating categories have 20+ films</p>';
        return;
    }
    
    let html = '';
    availableRatings.forEach(ratingInfo => {
        const starText = ratingInfo.rating === 0.5 ? '½' : '★'.repeat(Math.floor(ratingInfo.rating)) + (ratingInfo.rating % 1 ? '½' : '');
        const ratingLabel = ratingInfo.rating === 5 ? '5★ Films' : `${starText}+ Films`;
        html += `
            <button class="rating-btn" onclick="predictFromLetterboxd(${ratingInfo.rating})">
                ${ratingLabel} (${ratingInfo.count > 50 ? 'latest 50' : ratingInfo.count} films)
            </button>
        `;
    });
    
    ratingButtonsDiv.innerHTML = html;
}

// Display fetched Letterboxd films
function displayLetterboxdFilms(matchedFilms, unmatchedFilms) {
    const filmsDiv = document.getElementById('letterboxdFilms');
    
    let html = '<h4>Fetched Films:</h4>';
    
    if (matchedFilms.length > 0) {
        html += '<h5>Matched Films (available for recommendations):</h5>';
        
        // Group by rating for display
        const filmsByRating = {};
        matchedFilms.forEach(film => {
            const rating = film.rating || 'Unrated';
            if (!filmsByRating[rating]) {
                filmsByRating[rating] = [];
            }
            filmsByRating[rating].push(film);
        });
        
        // Sort ratings in descending order
        const sortedRatings = Object.keys(filmsByRating).sort((a, b) => {
            if (a === 'Unrated') return 1;
            if (b === 'Unrated') return -1;
            return parseFloat(b) - parseFloat(a);
        });
        
        sortedRatings.forEach(rating => {
            const starText = rating === 'Unrated' ? rating : 
                           rating === '0.5' ? '½★' : 
                           '★'.repeat(Math.floor(parseFloat(rating))) + (parseFloat(rating) % 1 ? '½' : '');
            
            html += `<div class="rating-group">
                <h6>${starText} (${filmsByRating[rating].length} films)</h6>`;
            
            filmsByRating[rating].slice(0, 10).forEach(film => {
                html += `<div class="letterboxd-film matched">✓ ${film.title} (ID: ${film.movieId})</div>`;
            });
            
            if (filmsByRating[rating].length > 10) {
                html += `<div class="film-count">... and ${filmsByRating[rating].length - 10} more</div>`;
            }
            
            html += '</div>';
        });
    }
    
    if (unmatchedFilms.length > 0) {
        html += '<h5>Unmatched Films:</h5>';
        unmatchedFilms.slice(0, 10).forEach(film => {
            const ratingText = film.rating ? ` (${film.rating}★)` : '';
            html += `<div class="letterboxd-film unmatched">✗ ${film.title}${ratingText}</div>`;
        });
        
        if (unmatchedFilms.length > 10) {
            html += `<div class="film-count">... and ${unmatchedFilms.length - 10} more unmatched films</div>`;
        }
    }
    
    filmsDiv.innerHTML = html;
    filmsDiv.style.display = 'block';
}

// Predict from Letterboxd films with optional rating filter
async function predictFromLetterboxd(rating = null) {
    const matchedIds = letterboxdClient.getMatchedIds(rating);
    
    if (!matchedIds || matchedIds.length === 0) {
        const ratingText = rating ? `${rating}+ star` : 'matched';
        alert(`No ${ratingText} films available for recommendations`);
        return;
    }
    
    try {
        // Disable all rating buttons during prediction
        const ratingButtons = document.querySelectorAll('.rating-btn');
        ratingButtons.forEach(btn => {
            btn.disabled = true;
            if (btn.onclick && btn.onclick.toString().includes(rating || 'predictFromLetterboxd()')) {
                btn.textContent = btn.textContent.replace(/\(.*\)/, '(Getting Recommendations...)');
            }
        });
        
        // Use the matched IDs for prediction
        const recommendations = await generateRecommendations(matchedIds);
        
        const ratingText = rating ? ` (${rating}+ star films)` : ' (Recent films)';
        displayRecommendations(recommendations, 'letterboxdRecommendations', ratingText);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Prediction error: ${error.message}`);
    } finally {
        // Re-enable all rating buttons
        const ratingButtons = document.querySelectorAll('.rating-btn');
        ratingButtons.forEach(btn => {
            btn.disabled = false;
        });
        
        // Restore original button text by re-displaying options
        const availableRatings = letterboxdClient.getAvailableRatingOptions();
        displayRecommendationOptions(availableRatings);
    }
}

// Main predict function
async function predict() {
    if (!session || !metadata) {
        alert('Model not loaded yet!');
        return;
    }
    
    const movieInput = document.getElementById('movieInput').value.trim();
    if (!movieInput) {
        alert('Please enter movie titles or IDs');
        return;
    }
    
    try {
        // Parse input - handle both titles and IDs
        const inputItems = movieInput.split(',').map(item => item.trim()).filter(item => item);
        
        if (inputItems.length === 0) {
            alert('Please enter valid movie titles or IDs');
            return;
        }
        
        document.getElementById('predictBtn').disabled = true;
        document.getElementById('predictBtn').textContent = 'Processing...';
        
        const { movieIds, matchedMovies, unmatchedItems } = parseMovieInput(inputItems);
        
        if (movieIds.length === 0) {
            alert('No valid movies found. Please check your input.');
            displayManualMatchedMovies([], unmatchedItems);
            return;
        }
        
        // Display matched/unmatched movies
        displayManualMatchedMovies(matchedMovies, unmatchedItems);
        
        document.getElementById('predictBtn').textContent = 'Predicting...';
        
        const recommendations = await generateRecommendations(movieIds);
        displayRecommendations(recommendations, 'recommendations', ` (from ${movieIds.length} movies)`);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Prediction error: ${error.message}`);
    } finally {
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('predictBtn').textContent = 'Get Recommendations';
    }
}

// Parse movie input - handle both titles and IDs
function parseMovieInput(inputItems) {
    const movieIds = [];
    const matchedMovies = [];
    const unmatchedItems = [];
    
    inputItems.forEach(item => {

        // It's a movie title - try to match
        const normalizedTitle = normalizeTitle(item, false); // Keep year first
        let movieId = movieTitles[normalizedTitle];
        
        if (!movieId) {
            // Try without year
            const normalizedTitleNoYear = normalizeTitle(item, true);
            movieId = movieTitles[normalizedTitleNoYear];
        }
        
        if (movieId) {
            movieIds.push(movieId);
            matchedMovies.push({
                input: item,
                movieId: movieId,
                title: movieIdToName[movieId.toString()],
                type: 'Title'
            });
        } else {
            unmatchedItems.push({ input: item, reason: 'Movie title not found in database' });
        }
    });
    
    return { movieIds, matchedMovies, unmatchedItems };
}

// Display matched and unmatched movies for manual input
function displayManualMatchedMovies(matchedMovies, unmatchedItems) {
    const matchedDiv = document.getElementById('manualMatchedMovies');
    
    if (matchedMovies.length === 0 && unmatchedItems.length === 0) {
        matchedDiv.style.display = 'none';
        return;
    }
    
    let html = '<h4>Input Processing Results:</h4>';
    
    if (matchedMovies.length > 0) {
        html += '<h5>Matched Movies:</h5>';
        matchedMovies.forEach(movie => {
            const typeLabel = movie.type === 'ID' ? 'ID' : 'Title';
            html += `
                <div class="letterboxd-film matched">
                    ✓ ${movie.input} → ${movie.title} (ID: ${movie.movieId}) [${typeLabel}]
                </div>
            `;
        });
    }
    
    if (unmatchedItems.length > 0) {
        html += '<h5>Unmatched Items:</h5>';
        unmatchedItems.forEach(item => {
            html += `
                <div class="letterboxd-film unmatched">
                    ✗ ${item.input} - ${item.reason}
                </div>
            `;
        });
    }
    
    matchedDiv.innerHTML = html;
    matchedDiv.style.display = 'block';
}

// Generate recommendations based on movie IDs
async function generateRecommendations(movieIds) {
    // Prepare input sequence with LEFT-PADDING like Python
    const maxLen = metadata.model_params.max_len;
    const inputSequence = new Array(maxLen).fill(0);
    
    // Left-pad: put movies at the END of the sequence
    const startIndex = Math.max(0, maxLen - movieIds.length);
    for (let i = 0; i < Math.min(movieIds.length, maxLen); i++) {
        inputSequence[startIndex + i] = movieIds[i];
    }
    
    console.log('Input movies:', movieIds.map(id => `${id}: ${getMovieName(id)}`));
    
    // Create tensor - use regular Int32Array instead of BigInt64Array
    const inputTensor = new ort.Tensor('int64', new BigInt64Array(inputSequence.map(x => BigInt(x))), [1, maxLen]);
        
    // Run inference
    const results = await session.run({ input_sequence: inputTensor });
    const outputData = results.output_logits.data;
        
    // Get top recommendations
    return getTopRecommendations(outputData, movieIds, 10);
}

// Get top K recommendations
function getTopRecommendations(outputData, watchedMovies, topK) {
    // Convert to regular array and get movie scores
    const scores = Array.from(outputData);
    const watchedSet = new Set(watchedMovies);
    
    // Create (movieId, score) pairs, excluding watched movies and movie ID 0 (padding)
    const movieScores = [];
    for (let i = 1; i < scores.length; i++) {
        if (!watchedSet.has(i)) {
            movieScores.push({ movieId: i, score: scores[i] });
        }
    }
    
    // Sort by score descending and take top K
    movieScores.sort((a, b) => b.score - a.score);
        
    return movieScores.slice(0, topK);
}

// Display recommendations in the UI
function displayRecommendations(recommendations, containerId, sourceText = '') {
    const recDiv = document.getElementById(containerId);
    
    if (recommendations.length === 0) {
        recDiv.innerHTML = '<p>No recommendations found.</p>';
        return;
    }
    
    let html = `<h4>Top Recommendations${sourceText}:</h4>`;
    recommendations.forEach((rec, index) => {
        const movieName = getMovieName(rec.movieId);
        html += `
            <div class="movie-item">
                <div class="movie-title">#${index + 1} - ${movieName}</div>
                <div class="movie-id">Movie ID: ${rec.movieId}</div>
                <div class="movie-score">Confidence Score: ${rec.score.toFixed(4)}</div>
            </div>
        `;
    });
    
    recDiv.innerHTML = html;
}

// Start initialization when page loads
window.addEventListener('load', init);