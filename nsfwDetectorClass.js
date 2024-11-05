class LRUCache {
    constructor(options = {}) {
        this.max = options.max || 100;
        this.maxAge = options.maxAge || 1000 * 60 * 60; // 1 hour default
        this.cache = new Map();
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;

        // Check if expired
        if (Date.now() - item.timestamp > this.maxAge) {
            this.cache.delete(key);
            return null;
        }

        // Refresh item's position in cache
        this.cache.delete(key);
        this.cache.set(key, item);
        
        return item.value;
    }

    set(key, value) {
        // Remove oldest if at max capacity
        if (this.cache.size >= this.max) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        // Compress confidence scores to 2 decimal places
        if (value.results) {
            value.results = value.results.map(r => ({
                ...r,
                probability: Math.round(r.probability * 100) / 100
            }));
        }

        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    delete(key) {
        this.cache.delete(key);
    }

    clear() {
        this.cache.clear();
    }

    get size() {
        return this.cache.size;
    }
}

class NsfwClassifier {
    constructor() {
        this.model = null;
        this.MODEL_URL = './models/mobilenet_v2/model.json';
        this.classes = ['drawing', 'hentai', 'neutral', 'porn', 'sexy'];
    }

    async loadModel() {
        if (!this.model) {
            this.model = await tf.loadLayersModel(this.MODEL_URL);
        }
        return this.model;
    }

    async classifyImage(imageElement) {
        const model = await this.loadModel();
        const predictions = await model.predict(
            tf.tidy(() => {
                return tf.browser.fromPixels(imageElement)
                    .resizeBilinear([224, 224])
                    .toFloat()
                    .div(tf.scalar(255))
                    .expandDims();
            })
        );
        const result = predictions.dataSync();
        predictions.dispose();
        return result;
    }

    async isNsfw(imageUrl) {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = imageUrl;

        await new Promise((resolve, reject) => {
            img.onload = resolve;
            img.onerror = reject;
        });

        const predictions = await this.classifyImage(img);
        const results = this.classes.map((className, index) => ({
            className,
            probability: predictions[index]
        }));

        // Sort results by probability in descending order
        results.sort((a, b) => b.probability - a.probability);

        // Determine if the image is NSFW based on 'hentai', 'porn', or 'sexy' probabilities
        const isNSFW = predictions[1] > 0.2 || predictions[3] > 0.2 || predictions[4] > 0.2;

        const result = {
            isNSFW,
            results
        };
        return result;
    }
}

class NsfwTextClassifier {
    constructor() {
        this.model = null;
        this.tokenizer = null;
        this.MODEL_URL = './models/tfjs_model_nsfw_classifier/model.json';
        this.TOKENIZER_URL = './models/nsfw_classifier_tokenizer.json';
        this.maxSequenceLength = 50;
    }

    async loadModel() {
        if (!this.model) {
            this.model = await tf.loadLayersModel(this.MODEL_URL);
            const tokenizerResponse = await fetch(this.TOKENIZER_URL);
            this.tokenizer = await tokenizerResponse.json();
        }
        return this.model;
    }

    tokenizeText(text) {
        // Convert text to lowercase and split into words
        const words = text.toLowerCase().split(/\s+/);
        
        // Convert words to token indices
        const sequence = words.map(word => this.tokenizer[word] || this.tokenizer['<OOV>']);
        
        // Pad sequence to fixed length
        const padded = sequence.slice(0, this.maxSequenceLength);
        while (padded.length < this.maxSequenceLength) {
            padded.push(0);  // Padding token
        }
        
        return padded;
    }

    async classifyText(prompt, negativePrompt = '') {
        const model = await this.loadModel();
        
        // Tokenize both prompts
        const promptTokens = this.tokenizeText(prompt);
        const negPromptTokens = this.tokenizeText(negativePrompt);
        
        // Convert to tensors
        const promptTensor = tf.tensor2d([promptTokens], [1, this.maxSequenceLength]);
        const negPromptTensor = tf.tensor2d([negPromptTokens], [1, this.maxSequenceLength]);
        
        // Get prediction
        const prediction = await model.predict([promptTensor, negPromptTensor]);
        const score = prediction.dataSync()[0];
        
        // Cleanup
        promptTensor.dispose();
        negPromptTensor.dispose();
        prediction.dispose();
        
        return {
            isNSFW: score > 0.5,
            score: score
        };
    }
}

class NsfwDetector {
    constructor() {
        this.nsfwClassifier = new NsfwClassifier();
        this.textClassifier = new NsfwTextClassifier();
        
        // Add cache with size limit to prevent memory issues
        this.cache = new LRUCache({
            max: 500,              // Store more results
            maxAge: 1000 * 60 * 60 // 1 hour cache
        });
        this.maxCacheSize = 100; // Limit cache to last 100 items
        this.cacheDuration = 5 * 60 * 1000; // 5 minutes in milliseconds
        
        // Fix: Assign arrays to class properties using 'this'
        this.nsfwKeywords = [
            "hardcore","kiss","kissing", "obscene", "nude", "nudity", "naked", "sensual", "provocative", 
            "suggestive", "fetish", "kink", "voyeur", "seductive", "sexual", "lustful", 
            "sultry", "risque", "taboo", "voyeuristic", "undressed", "unclothed", "bare", 
            "exposed", "intimate", "raunchy", "strip", "stripping", "stripped", "undressing", 
            "breasts", "boobs", "tits", "nipples", "genitals", "vagina", "penis", "testicles", 
            "scrotum", "pubic", "buttocks", "ass", "groin", "crotch", "thighs", "hips", 
            "threesome", "orgy", "orgasm", "intercourse", "masturbation", "foreplay", 
            "oral", "blowjob", "ejaculation", "penetration", "incest", "molest", 
            "molestation", "rape", "raping", "pedo", "pedophile", "child abuse", 
            "child pornography", "underage", "loli", "cp", "young girl", "young boy", 
            "schoolgirl", "schoolboy", "teen", "minor", "jailbait", "child", "infant", 
            "baby", "toddler", "pubescent", "prepubescent", "underaged", "innocent", 
            "ch1ld", "ch!ld", "fet1sh", "masturb@t1on", "int1m@te", "h@rdcore", 
            "rap3", "mol3st", "undr3ssed", "int1m@te", "ej@cul@tion", "p3n!s", "gen!tal", 
            "bre@st", "b00bs", "n!pples", "h1ps", "b0dy", "v@gina", "sc@ntily", 
            "sl@ve", "dom!nant", "b0dy", "t!ts", "c0rs3t", "br@", "p@nty", "br@less", 
            "exposed", "corset", "lingerie", "thong", "panties", "sheer", "see-through", 
            "topless", "braless", "stripper", "strippers", "lap dance", "camgirl", "camgirls", 
            "dominatrix", "submissive", "degrading", "degrade", "abuse","abuser", "abusive", "shame", "shaming", 
            "bondage", "bdsm", "slave", "master", "mistress", "discipline", "spank", 
            "whip", "gag", "choke", "handcuff", "restraint", "leather", "latex", 
            "g-string", "garter", "fishnet", "stockings", "thigh-highs", "corset", 
            "dominant", "submissive", "roleplay", "sex", "hardcore sex", "softcore", 
            "soft porn", "hard porn", "cam", "webcam", "onlyfans", "fansly", "sex work", 
            "prostitute", "escort", "escort services", "hooker", "hooking", "massage parlor", 
            "happy ending", "escort services", "body rubs", "massage", "nuru", "erotica", 
            "bdsm", "domination", "submissive", "sadomasochism", "sadism", "masochism", 
            "gore", "snuff", "beheading", "brutality", "torture", "gruesome", "disgusting", 
            "grotesque", "bloody", "mutilation", "self-harm", "suicide", "killing", 
            "murder", "slaughter", "injure", "injury", "rape scene", "porn video", 
            "sex tape", "sex video", "creampie", "facial", "gagging", "rough sex", 
            "anal", "fisting", "gaping", "penetrate", "penile", "vaginal", "g-string", 
            "seduce", "seducing", "intercourse", "copulation", "intimacy", "intimate", 
            "intimately", "making out", "make out", "foreplay", "tease", "aroused", 
            "arousal", "horny", "desire", "kama sutra", "doggy style", "missionary", 
            "cowgirl", "reverse cowgirl", "69 position", "dominant position", "top position", 
            "bottom position", "sexual fantasy", "fantasy", "fetishist", "fetishistic", 
            "foot fetish", "lingerie model", "intimate apparel", "racy", "sizzling", 
            "tempting", "inviting", "pornstar", "centerfold", "hustler", "playboy", 
            "playmate", "penthouse", "eroticism", "suggestive", "naughty", "sultry", 
            "lustful", "flirty", "flirting", "provoking", "provocative", "tempting", 
            "temptress", "seduction", "vixen", "temptress", "desirable", "sexiness", 
            "alluring", "lust", "dirty talk", "sexy talk", "sexting", "dirty message", 
            "x-rated", "triple-x", "slut", "whore", "tramp", "harlot", "bimbo", 
            "swinger", "swinging", "swingers", "seduction", "flirt", "flirting", 
            "seductress", "lusting", "hot", "hottie", "sexual encounter", "hookup", 
            "sizzling hot", "heated", "arouse", "aroused", "horny", "seductive", 
            "come-hither", "racy", "rave", "body shots", "strip show", "peep show", 
            "sexual innuendo", "innuendo", "sexual overtones", "hot chick", "hot guy", 
            "hot girl", "playgirl", "adult movie", "adult site", "erotic movie", 
            "adult content", "bareback", "buttplug", "anal beads", "dildo", "vibrator", 
            "sex toy", "strap-on", "sex slave", "dominatrix", "latex","bra","bondage gear",
            "lingerie", "swimwear", "bikini","bathtub","sweaty","flesh body", "skin", "flesh","legs woman","plump woman","chubby woman body"
            // Add remaining NSFW keywords here
        ];
        
        this.under20Keywords = [
            "adolescent", "baby", "birthday party", "boy", "child", "classmate", "daycare", 
            "freshman", "girl", "high school", "infant", "junior", "kid", "kids", "kindergartener", 
            "little boy", "little girl", "middle school", "minor", "preschool", "primary school", 
            "pubescent", "schoolboy", "schoolgirl", "second grader", "senior", "sophomore", 
            "student", "teen", "teenager", "toddler", "under 20", "underage", "young adult", 
            "youngster", "youth"
        ];
        
        this.ageThreshold = 22;
        this.neutralThreshold = 0.9; // 90% threshold for neutral class

        // Image processing settings
        this.imageSettings = {
            maxSize: 512,          // Capped at 512
            progressiveSizes: [     
                256,              // Fast first pass
                512               // Final check if needed
            ],
            confidenceThresholds: {
                low: 0.5,
                high: 0.8         // Can be more strict with smaller sizes
            }
        };

        // Enhanced caching with metadata
        this.cache = new LRUCache({
            max: 500,              // Store more results
            maxAge: 1000 * 60 * 60 // 1 hour cache
        });

        // Add to class properties
        this.imagePreloadQueue = new Map();
        this.preloadLimit = 5;  // Limit concurrent preloads
    }

    async initialize() {
        await Promise.all([
            this.nsfwClassifier.loadModel(),
            this.textClassifier.loadModel(),
            faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
            faceapi.nets.ageGenderNet.loadFromUri('./models')
        ]);
    }

    containsKeywords(text) {
        if (!text) return false;
        const lowerText = text.toLowerCase();
        
        // Create regex patterns with word boundaries
        const matchWord = (keyword) => {
            const pattern = new RegExp(`\\b${keyword}\\b`, 'i');
            return pattern.test(lowerText);
        };

        return this.nsfwKeywords.some(matchWord) || 
               this.under20Keywords.some(matchWord);
    }

    convertHotpotLinkToS3(hotpotLink) {
        const match = hotpotLink.match(/\/art-generator\/(8-[\w\d]+)\?/);
        return match ? `https://hotpotmedia.s3.us-east-2.amazonaws.com/${match[1]}.png` : null;
    }

    async analyzeImage(imageUrl) {
        try {
            const result = await this.processImageProgressive(imageUrl);
            return result;
        } catch (error) {
            return { isNSFW: true, reason: 'Processing error' };
        }
    }

    async isNsfw(hotpotLink) {
        const cachedResult = this.getFromCache(hotpotLink);
        if (cachedResult) {
            if (cachedResult.isNSFW) {
                console.log('NSFW detected:', hotpotLink, 'Reason:', cachedResult.reason);
            }
            return cachedResult;
        }

        const url = new URL(hotpotLink);
        const title = url.searchParams.get("title");
    
        if (this.containsKeywords(title)) {
            const result = { isNSFW: true, reason: 'Keyword match' };
            this.addToCache(hotpotLink, result);
            console.log('NSFW detected:', hotpotLink, 'Reason:', result.reason);
            return result;
        }
    
        const textResult = await this.textClassifier.classifyText(title);
        if (textResult.isNSFW) {
            const result = { isNSFW: true, reason: 'Text classification' };
            this.addToCache(hotpotLink, result);
            console.log('NSFW detected:', hotpotLink, 'Reason:', result.reason);
            return result;
        }
    
        const imageUrl = this.convertHotpotLinkToS3(hotpotLink);
        if (!imageUrl) {
            return { isNSFW: false, reason: 'Link conversion failed' };
        }
    
        const result = await this.analyzeImage(imageUrl);
        const finalResult = result.isNSFW ? 
            { isNSFW: true, reason: 'Image classification' } : 
            { isNSFW: false, imageUrl: imageUrl };
            
        this.addToCache(hotpotLink, finalResult);
        if (finalResult.isNSFW) {
            console.log('NSFW detected:', hotpotLink, 'Reason:', finalResult.reason);
        }
        return finalResult;
    }

    getFromCache(key) {
        const cached = this.cache.get(key);
        if (!cached) return null;

        // Check if cache entry has expired
        if (Date.now() - cached.timestamp > this.cacheDuration) {
            this.cache.delete(key);
            return null;
        }

        return cached.result;
    }

    addToCache(key, result) {
        // Remove oldest entry if cache is full
        if (this.cache.size >= this.maxCacheSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        this.cache.set(key, {
            result,
            timestamp: Date.now()
        });
    }

    async loadImage(url) {
        // Check preload cache first
        const preloaded = this.imagePreloadQueue.get(url);
        if (preloaded) {
            try {
                const img = await preloaded.promise;
                this.imagePreloadQueue.delete(url);  // Cleanup
                return img;
            } catch (error) {
                this.imagePreloadQueue.delete(url);  // Cleanup on error
            }
        }

        // Fallback to regular loading
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            
            const timeout = setTimeout(() => {
                reject(new Error('Image loading timeout'));
            }, 10000); // 10 second timeout
            
            img.onload = () => {
                clearTimeout(timeout);
                resolve(img);
            };
            
            img.onerror = () => {
                clearTimeout(timeout);
                reject(new Error('Failed to load image'));
            };
            
            img.src = url + (url.includes('?') ? '&' : '?') + 'cache=' + Date.now();
        });
    }

    async detectAge(img) {
        const detections = await faceapi
            .detectAllFaces(img, new faceapi.TinyFaceDetectorOptions())
            .withAgeAndGender();

        if (detections.length > 0) {
            return Math.round(detections[0].age);
        }
        return null;
    }

    async processImageProgressive(imageUrl) {
        const cacheKey = `${imageUrl}_${this.imageSettings.maxSize}`;
        const cached = this.getFromCache(cacheKey);
        if (cached) return cached;

        let result = null;
        let confidence = 0;

        try {
            // Progressive resolution checking
            for (const size of this.imageSettings.progressiveSizes) {
                // Create an image element
                const img = await this.loadImage(imageUrl);
                
                // Resize image
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                // Calculate new dimensions
                let [width, height] = this.calculateDimensions(img, size);
                canvas.width = width;
                canvas.height = height;
                
                // Draw resized image
                ctx.drawImage(img, 0, 0, width, height);
                
                // Convert canvas to blob
                const blob = await new Promise(resolve => {
                    canvas.toBlob(resolve, 'image/jpeg', 0.9);
                });
                
                // Create URL from blob
                const resizedImageUrl = URL.createObjectURL(blob);
                
                // Process with NSFW classifier
                const currentResult = await this.nsfwClassifier.isNsfw(resizedImageUrl);
                
                // Clean up
                URL.revokeObjectURL(resizedImageUrl);
                
                confidence = Math.max(...currentResult.results.map(r => r.probability));
                
                if (currentResult.isNSFW && confidence > this.imageSettings.confidenceThresholds.high) {
                    result = { isNSFW: true, confidence, resolution: size };
                    break;
                }
                
                result = { 
                    isNSFW: currentResult.isNSFW, 
                    confidence,
                    resolution: size 
                };
            }

            this.addToCache(cacheKey, result);
            return result;

        } catch (error) {
            throw error;
        }
    }

    // Helper function to calculate dimensions
    calculateDimensions(img, maxSize) {
        const ratio = Math.min(maxSize / img.width, maxSize / img.height);
        return [
            Math.round(img.width * ratio),
            Math.round(img.height * ratio)
        ];
    }

    // Enhanced caching with compression
    addToCache(key, value) {
        // Compress confidence scores to 2 decimal places
        if (value.results) {
            value.results = value.results.map(r => ({
                ...r,
                probability: Math.round(r.probability * 100) / 100
            }));
        }

        this.cache.set(key, {
            ...value,
            cached: Date.now()
        });
    }

    // Preload method
    preloadImages(urls) {
        // Remove old preloads
        for (const [url, data] of this.imagePreloadQueue) {
            if (Date.now() - data.timestamp > 30000) {  // 30s timeout
                this.imagePreloadQueue.delete(url);
            }
        }

        // Add new URLs to preload queue
        urls.forEach(url => {
            if (!this.imagePreloadQueue.has(url) && 
                this.imagePreloadQueue.size < this.preloadLimit) {
                
                const img = new Image();
                img.crossOrigin = 'anonymous';
                
                const promise = new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => {
                        this.imagePreloadQueue.delete(url);
                        reject(new Error('Preload timeout'));
                    }, 30000);

                    img.onload = () => {
                        clearTimeout(timeout);
                        resolve(img);
                    };
                    
                    img.onerror = () => {
                        clearTimeout(timeout);
                        this.imagePreloadQueue.delete(url);
                        reject(new Error('Failed to preload'));
                    };
                });

                this.imagePreloadQueue.set(url, {
                    img,
                    promise,
                    timestamp: Date.now()
                });
                
                img.src = url + (url.includes('?') ? '&' : '?') + 'cache=' + Date.now();
            }
        });
    }
}

// Make both classes available globally
window.NsfwClassifier = NsfwClassifier;
window.NsfwDetector = NsfwDetector;
