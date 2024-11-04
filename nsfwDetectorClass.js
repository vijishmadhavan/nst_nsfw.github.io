class NsfwClassifier {
    constructor() {
        this.model = null;
        this.MODEL_URL = './models/mobilenet_v2/model.json';
        this.classes = ['drawing', 'hentai', 'neutral', 'porn', 'sexy'];
    }

    async loadModel() {
        if (!this.model) {
            console.log('Loading model from:', this.MODEL_URL);
            this.model = await tf.loadLayersModel(this.MODEL_URL);
            console.log('Model loaded:', this.model);
        }
        return this.model;
    }

    async classifyImage(imageElement) {
        const model = await this.loadModel();
        const tensor = tf.browser.fromPixels(imageElement)
            .resizeBilinear([224, 224]) // MobileNet V2 expects 224x224 images
            .toFloat()
            .div(tf.scalar(255))
            .expandDims();
        const predictions = await model.predict(tensor);
        return predictions.dataSync();
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
        console.log('NSFW detection result:', result);
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
            console.log('Loading text classification model...');
            this.model = await tf.loadLayersModel(this.MODEL_URL);
            const tokenizerResponse = await fetch(this.TOKENIZER_URL);
            this.tokenizer = await tokenizerResponse.json();
            console.log('Text classification model loaded');
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
            "lingerie", "swimwear", "bikini"
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
    }

    async initialize() {
        console.time('Total Model Loading Time');
        
        // Individual model timing
        console.log('Starting model loading...');
        
        const startTime = performance.now();
        
        try {
            await Promise.all([
                this.timeModelLoad('NSFW Classifier', () => this.nsfwClassifier.loadModel()),
                this.timeModelLoad('Text Classifier', () => this.textClassifier.loadModel()),
                this.timeModelLoad('Face Detector', () => faceapi.nets.tinyFaceDetector.loadFromUri('./models')),
                this.timeModelLoad('Age Gender Net', () => faceapi.nets.ageGenderNet.loadFromUri('./models'))
            ]);
            
            const endTime = performance.now();
            console.log(`Total initialization time: ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
        } catch (error) {
            console.error('Error during initialization:', error);
        }
        
        console.timeEnd('Total Model Loading Time');
    }

    async timeModelLoad(modelName, loadFunction) {
        const start = performance.now();
        await loadFunction();
        const end = performance.now();
        console.log(`${modelName} loaded in ${((end - start) / 1000).toFixed(2)} seconds`);
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
        const nsfwResult = await this.nsfwClassifier.isNsfw(imageUrl);

        if (nsfwResult.isNSFW) {
            // If already NSFW, skip age detection
            return {
                isNSFW: true,
                age: null,
                nsfwResults: nsfwResult.results
            };
        }

        // Check if the neutral class probability is above the threshold
        const neutralProb = nsfwResult.results.find(r => r.className === 'neutral')?.probability || 0;
        if (neutralProb > this.neutralThreshold) {
            // If highly neutral, skip age detection
            return {
                isNSFW: false,
                age: null,
                nsfwResults: nsfwResult.results
            };
        }

        // Only load the image and perform age detection if not NSFW and not highly neutral
        const img = await this.loadImage(imageUrl);
        const ageResult = await this.detectAge(img);

        const isNSFW = nsfwResult.isNSFW || (ageResult !== null && ageResult < this.ageThreshold);

        return {
            isNSFW,
            age: ageResult,
            nsfwResults: nsfwResult.results
        };
    }

    async isNsfw(hotpotLink) {
        const url = new URL(hotpotLink);
        const title = url.searchParams.get("title");
    
        // First check: Keywords
        if (this.containsKeywords(title)) {
            console.log("NSFW or under-20 content detected in keywords");
            return { isNSFW: true, reason: 'Keyword match' };
        }
    
        // Second check: Text classification
        const textResult = await this.textClassifier.classifyText(title);
        if (textResult.isNSFW) {
            console.log("NSFW content detected by text classifier");
            return { isNSFW: true, reason: 'Text classification' };
        }
    
        // Third check: Image analysis
        const imageUrl = this.convertHotpotLinkToS3(hotpotLink);
        if (!imageUrl) {
            console.error("Failed to convert Hotpot link");
            return { isNSFW: false, reason: 'Link conversion failed' };
        }
    
        const result = await this.analyzeImage(imageUrl);
        return result.isNSFW ? 
            { isNSFW: true, reason: 'Image classification' } : 
            { isNSFW: false, imageUrl: imageUrl };
    }
    

    async loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
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
}

// Make both classes available globally
window.NsfwClassifier = NsfwClassifier;
window.NsfwDetector = NsfwDetector;
