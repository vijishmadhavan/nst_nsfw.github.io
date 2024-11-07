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
            try {
                // Try to load from IndexedDB
                this.model = await tf.loadLayersModel('indexeddb://nsfw-classifier-model');
                // Model loaded from IndexedDB
            } catch (e) {
                // Load model from URL
                this.model = await tf.loadLayersModel(this.MODEL_URL);
                // Save to IndexedDB for future use
                await this.model.save('indexeddb://nsfw-classifier-model');
            }
        }
        return this.model;
    }

    async classifyImage(imageElement) {
        const model = await this.loadModel();
        let predictions;
        try {
            predictions = await model.predict(
                tf.tidy(() => {
                    return tf.browser.fromPixels(imageElement)
                        .resizeBilinear([224, 224])
                        .toFloat()
                        .div(tf.scalar(255))
                        .expandDims();
                })
            );
            const result = predictions.dataSync();
            return result;
        } finally {
            if (predictions) predictions.dispose();
        }
    }

    async isNsfw(imageElement) {
        const predictions = await this.classifyImage(imageElement);
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
            try {
                // Try to load model from IndexedDB
                this.model = await tf.loadLayersModel('indexeddb://nsfw-text-classifier-model');
            } catch (e) {
                // Load model from URL
                this.model = await tf.loadLayersModel(this.MODEL_URL);
                // Save model to IndexedDB
                await this.model.save('indexeddb://nsfw-text-classifier-model');
            }
            // Load tokenizer
            if (!this.tokenizer) {
                const tokenizerResponse = await fetch(this.TOKENIZER_URL);
                this.tokenizer = await tokenizerResponse.json();
            }
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
            padded.push(0); // Padding token
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
        let prediction;
        try {
            prediction = await model.predict([promptTensor, negPromptTensor]);
            const score = prediction.dataSync()[0];

            return {
                isNSFW: score > 0.5,
                score: score
            };
        } finally {
            promptTensor.dispose();
            negPromptTensor.dispose();
            if (prediction) prediction.dispose();
        }
    }
}

class NsfwDetector {
    constructor() {
        this.nsfwClassifier = new NsfwClassifier();
        this.textClassifier = new NsfwTextClassifier();

        this.maxCacheSize = 500; // Adjusted cache size
        this.cacheDuration = 5 * 60 * 1000; // 5 minutes in milliseconds

        this.cache = new LRUCache({
            max: this.maxCacheSize,
            maxAge: this.cacheDuration
        });

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
            "lingerie", "swimwear", "bikini","bathtub","sweaty"
            // Add remaining NSFW keywords here
        ];

        // Under 20 keywords (list truncated for brevity)
        this.under20Keywords = [
            "adolescent", "baby", "birthday party", "boy", "child", "classmate", "daycare", 
            "freshman", "girl", "high school", "infant", "junior", "kid", "kids", "kindergartener", 
            "little boy", "little girl", "middle school", "minor", "preschool", "primary school", 
            "pubescent", "schoolboy", "schoolgirl", "second grader", "senior", "sophomore", 
            "student", "teen", "teenager", "toddler", "under 20", "underage", "young adult", 
            "youngster", "youth"
        ];
        // Precompile regex patterns
        this.nsfwKeywordRegex = new RegExp(`\\b(${this.nsfwKeywords.join('|')})\\b`, 'i');
        this.under20KeywordRegex = new RegExp(`\\b(${this.under20Keywords.join('|')})\\b`, 'i');

        this.imageSettings = {
            maxSize: 512,          // Capped at 512
            progressiveSizes: [256, 512], // Adjusted sizes
            confidenceThresholds: {
                256: 0.8,
                512: 0.6
            }
        };

        // Preload queue
        this.imagePreloadQueue = new Map();
        this.preloadLimit = 5;  // Limit concurrent preloads

        this.faceModelsLoaded = false; // For lazy loading face models
    }

    async initialize() {
        // Load models
        await Promise.all([
            this.nsfwClassifier.loadModel(),
            this.textClassifier.loadModel()
        ]);
    }

    containsKeywords(text) {
        if (!text) return false;
        const lowerText = text.toLowerCase();

        return this.nsfwKeywordRegex.test(lowerText) || this.under20KeywordRegex.test(lowerText);
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
        // Check cache first
        const cachedResult = this.getFromCache(hotpotLink);
        if (cachedResult) {
            return cachedResult;
        }

        const url = new URL(hotpotLink);
        const title = url.searchParams.get("title");

        if (this.containsKeywords(title)) {
            const result = { isNSFW: true, reason: 'Keyword match' };
            this.addToCache(hotpotLink, result);
            // Log the NSFW detection
            console.log(`NSFW detected: ${hotpotLink}`);
            return result;
        }

        const textResult = await this.textClassifier.classifyText(title);

        if (textResult.isNSFW) {
            const result = { isNSFW: true, reason: 'Text classification' };
            this.addToCache(hotpotLink, result);
            // Log the NSFW detection
            console.log(`NSFW detected: ${hotpotLink}`);
            return result;
        }

        const imageUrl = this.convertHotpotLinkToS3(hotpotLink);
        if (!imageUrl) {
            const result = { isNSFW: false, reason: 'Link conversion failed' };
            this.addToCache(hotpotLink, result);
            return result;
        }

        const result = await this.analyzeImage(imageUrl);

        const finalResult = result.isNSFW ?
            { isNSFW: true, reason: 'Image classification' } :
            { isNSFW: false, imageUrl: imageUrl };

        this.addToCache(hotpotLink, finalResult);

        if (finalResult.isNSFW) {
            // Log the NSFW detection
            console.log(`NSFW detected: ${imageUrl}`);
        }

        return finalResult;
    }

    getFromCache(key) {
        const cached = this.cache.get(key);
        if (!cached) return null;

        return cached; // LRUCache handles expiration
    }

    addToCache(key, result) {
        this.cache.set(key, result);
    }

    async loadImage(url) {
        try {
            const response = await fetch(url, { mode: 'cors' });
            const blob = await response.blob();
            const imgBitmap = await createImageBitmap(blob);
            return imgBitmap;
        } catch (error) {
            throw error;
        }
    }

    async processImageProgressive(imageUrl) {
        const progressiveSizes = this.imageSettings.progressiveSizes;
        const confidenceThresholds = this.imageSettings.confidenceThresholds;

        let result = null;

        try {
            for (const size of progressiveSizes) {
                const img = await this.loadImage(imageUrl);
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');

                let [width, height] = this.calculateDimensions(img, size);
                canvas.width = width;
                canvas.height = height;

                // Draw the image onto the canvas
                ctx.drawImage(img, 0, 0, width, height);

                // Convert canvas to blob
                const blob = await new Promise(resolve =>
                    canvas.toBlob(resolve, 'image/jpeg')
                );

                // Create ImageBitmap from blob
                const resizedImg = await createImageBitmap(blob);

                const currentResult = await this.nsfwClassifier.isNsfw(resizedImg);

                const confidence = Math.max(...currentResult.results.map(r => r.probability));

                // Early return if we're confident enough at current resolution
                if (confidence > confidenceThresholds[size]) {
                    result = {
                        isNSFW: currentResult.isNSFW,
                        confidence,
                        resolution: size
                    };
                    break;
                }

                result = {
                    isNSFW: currentResult.isNSFW,
                    confidence,
                    resolution: size
                };
            }

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

                const promise = fetch(url, { mode: 'cors' })
                    .then(response => response.blob())
                    .then(blob => createImageBitmap(blob))
                    .then(imgBitmap => ({ imgBitmap, timestamp: Date.now() }))
                    .catch(error => {
                        this.imagePreloadQueue.delete(url);
                    });

                this.imagePreloadQueue.set(url, {
                    promise,
                    timestamp: Date.now()
                });
            }
        });
    }
}

// Make classes available globally
window.NsfwClassifier = NsfwClassifier;
window.NsfwTextClassifier = NsfwTextClassifier;
window.NsfwDetector = NsfwDetector;


