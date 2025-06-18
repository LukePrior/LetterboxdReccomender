class LetterboxdClient {
    constructor() {
        this.matchedIds = [];
        this.movieTitles = {};
        this.movieIdToName = null;
        this.allMatchedFilms = []; // Store all matched films with ratings
    }

    // Initialize with movie data
    setMovieData(movieIdToName, movieTitles) {
        this.movieIdToName = movieIdToName;
        this.movieTitles = movieTitles;
    }

    // Normalize title function matching updated implementation
    normalizeTitle(title, removeYear = true) {
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

    // Title similarity function with stricter matching
    titlesSimilar(title1, title2, threshold = 0.7) {
        // Simple Jaccard similarity using words
        const words1 = new Set(title1.split(' ').filter(word => word.length > 1));
        const words2 = new Set(title2.split(' ').filter(word => word.length > 1));
        
        if (words1.size === 0 || words2.size === 0) {
            return false;
        }
        
        // For very short titles, require exact match
        if (words1.size <= 2 && words2.size <= 2) {
            return title1 === title2;
        }
        
        const intersection = new Set([...words1].filter(word => words2.has(word)));
        const union = new Set([...words1, ...words2]);
        
        const similarity = union.size > 0 ? intersection.size / union.size : 0;
        
        return similarity >= threshold;
    }

    // Film matching function with updated logic
    findMovieIdByTitle(filmTitle) {
        if (!this.movieTitles || !this.movieIdToName) return null;
        
        
        // Step 1: Convert to lowercase and trim (keeping years) and check for direct match
        const normalizedWithYear = this.normalizeTitle(filmTitle, false);
        
        if (normalizedWithYear && normalizedWithYear.trim() !== '') {
            if (normalizedWithYear in this.movieTitles) {
                return this.movieTitles[normalizedWithYear];
            }
        }
        
        // Step 2: Remove year and check for direct match
        const normalizedWithoutYear = this.normalizeTitle(filmTitle, true);
        
        if (normalizedWithoutYear && normalizedWithoutYear.trim() !== '') {
            if (normalizedWithoutYear in this.movieTitles) {
                return this.movieTitles[normalizedWithoutYear];
            }
        }
        
        // Step 3: Fuzzy matching against titles without years
        for (const [dbTitle, movieId] of Object.entries(this.movieTitles)) {
            // Only do fuzzy matching against titles without years (no parentheses)
            if (!dbTitle.includes('(') && this.titlesSimilar(normalizedWithoutYear, dbTitle, 0.7)) {
                return movieId;
            }
        }
        
        return null;
    }

    // Fetch user films from Letterboxd
    async fetchUserFilms(username) {
        const films = [];
        const maxPages = 3; // Limit to first 3 pages to get ~200 films
        
        for (let page = 1; page <= maxPages; page++) {
            try {
                const url = `https://letterboxd.com/${username}/films/by/rated-date/page/${page}/`;
                
                // Use a CORS proxy service
                const proxyUrl = `https://corsproxy.io/?url=${encodeURIComponent(url)}`;
                
                const response = await fetch(proxyUrl);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const html = await response.text();
                
                // Parse HTML
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                
                // Find film elements (<li class="poster-container">)
                const filmElements = doc.querySelectorAll('li.poster-container');

                if (filmElements.length === 0) {
                    console.warn(`No films found on page ${page}`);
                    break; // Stop if no films found
                }

                // Extract film data
                for (const filmElement of filmElements) {
                    const img = filmElement.querySelector('img');
                    let title = img ? img.getAttribute('alt') : null;

                    // Get year from title if available
                    let year = null;
                    const div = filmElement.querySelector('div');
                    let slug = div ? div.getAttribute('data-film-slug') : null;
                    if (slug && slug.endsWith('-1')) {
                        slug = slug.slice(0, -2);
                    }
                    let slugMatch = slug ? slug.match(/-(\d{4})$/) : null;
                    if (slugMatch) {
                        slug = slug.slice(0, -slugMatch[0].length);
                        year = slugMatch[1];
                    }

                    if (year > 2023) {
                        continue; // Skip films with future years
                    }

                    // Add year to title if not already present
                    if (title && year && !title.includes(year)) {
                        title = `${title} (${year})`;
                    }

                    // Get rating if available
                    let rating = null;
                    const ratingSpan = filmElement.querySelector('span.rating');
                    if (ratingSpan) {
                        const ratingText = ratingSpan.textContent.trim();
                        const starMatch = ratingText.match(/★/g);
                        let stars = starMatch ? starMatch.length : 0;
                        if (ratingText.includes('½')) {
                            stars += 0.5;
                        }
                        rating = stars > 0 ? stars : null; // Only include if rating is greater than 0
                    }

                    // Add to films array
                    films.push({
                        title,
                        rating
                    });
                }
                
                // Add delay between requests to be respectful
                await new Promise(resolve => setTimeout(resolve, 500));
                
            } catch (error) {
                console.error(`Error fetching page ${page}:`, error);
                if (page === 1) {
                    throw error; // If first page fails, throw error
                }
                break; // For subsequent pages, just stop trying
            }
        }
        return films.slice(0, 200); // Ensure we don't exceed 200 films
    }

    // Get films by rating with minimum count check (includes films rated at this level AND higher)
    getFilmsByRating(targetRating, minCount = 20) {
        const ratedFilms = this.allMatchedFilms.filter(film => 
            film.rating && film.rating >= targetRating
        );
        
        if (ratedFilms.length >= minCount) {
            return ratedFilms.slice(0, 50).map(film => film.movieId);
        }
        
        return null; // Not enough films with this rating or higher
    }

    // Get available rating options (ratings with at least minCount films at that level or higher)
    getAvailableRatingOptions(minCount = 20) {
        const targetRatings = [5, 4.5, 4, 3.5, 3];
        const availableRatings = [];
        
        targetRatings.forEach(rating => {
            // Count films at this rating level or higher
            const count = this.allMatchedFilms.filter(film => 
                film.rating && film.rating >= rating
            ).length;
            
            if (count >= minCount) {
                availableRatings.push({ rating, count });
            }
        });
        
        return availableRatings;
    }

    // Process and match films with the database
    async processUserFilms(username) {
        const films = await this.fetchUserFilms(username);
        
        if (films.length === 0) {
            throw new Error('No films found for this user');
        }
        
        // Match films with our database - mirroring Python logic
        const matchedFilms = [];
        const unmatchedFilms = [];
        this.allMatchedFilms = []; // Reset all matched films
        this.matchedIds = [];
        
        let processedCount = 0;
        let matchedCount = 0;
        
        for (const film of films) {
            processedCount++;
            
            const matchedMovieId = this.findMovieIdByTitle(film.title);
            
            if (matchedMovieId) {
                const matchedFilm = { ...film, movieId: matchedMovieId };
                matchedFilms.push(matchedFilm);
                this.allMatchedFilms.push(matchedFilm);
                this.matchedIds.push(matchedMovieId);
                matchedCount++;
            } else {
                unmatchedFilms.push(film);
            }
        }
        
        // Reverse the order and limit to 50 for default (like Python)
        this.matchedIds = this.matchedIds.slice().slice(0, 50).reverse();
        
        return {
            matchedFilms,
            unmatchedFilms,
            totalFilms: films.length,
            matchedIds: this.matchedIds,
            availableRatings: this.getAvailableRatingOptions()
        };
    }

    // Get the matched movie IDs for recommendations (default or by rating)
    getMatchedIds(rating = null) {
        if (rating === null) {
            return this.matchedIds; // Return default top 50
        }
        
        return this.getFilmsByRating(rating);
    }
}

// Export for use in main script
window.LetterboxdClient = LetterboxdClient;