import numpy as np
import pickle
from os.path import join as pjoin

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'face', 'hand', 'leg', 'kuckles', 'thumb', 'index', 'finger', 'middle', 'ring', 'jaw')

Obj_List = (
    "teapot", "lid", "kettle", "sugar", "can", "spoon", "cup", "tea", "box", 
    "bag", "wrapper", "cream", "trash", "kitchen", "towel", "piece", "lime", "wooden", 
    "citrus", "reamer", "orange", "instant", "noodle", "container", "veggie", "oil", 
    "packet", "fork", "chopsticks", "tissue", "burger", "wrapper", "pile", "fried", 
    "chicken", "thigh", "bigmac", "ketchup", "bottle", "dipping", "sauce", "pudding", 
    "jar", "cloth", "rubber", "band", "plate", "bowl", "straw", "soda", "sanitizer", 
    "napkin", "toaster", "bread", "metal", "twist", "bag", "peanut", "butter", "knife", 
    "can", "spam", "cutting", "board", "tongs", "spatula", "holder", "egg", "carton", 
    "shell", "grinder", "salt", "shaker", "lettuce", "apple", "faucet", "washing", 
    "colander", "salad", "fruit", "sheath", "peeler", "banana", "nut", "mortar", "pestle", 
    "cheese", "grater", "grape", "dressing", "mop", "sieve", "mixing", "ziploc", 
    "measure", "cup", "dough", "scraper", "baking", "sheet", "rolling", "pin", "tray", 
    "sharp", "oven", "sawing", "cake", "stand", "piping", "book", "note", "belt",
    "pencil", "marker", "highlighter", "eraser", "correction", "notebook", "envelope", 
    "clamp", "letter", "opener", "stapler", "stamp", "paper", "laptop", "power", 
    "socket", "type-c", "usb", "audio", "mouse", "headphone", "charging", "tablet", 
    "digital", "spray", "microfiber", "phone", "sim", "watch", "earphone", "gopro", 
    "selfie", "tripod", "present", "wrapping", "elephant", "gift", "ribbon", "ruler", 
    "tape", "scissors", "lamp", "stone", "sand", "carving", "chisel", "sharpening", 
    "tool", "bench", "wood", "clamp", "handsaw", "hammer", "nail", "screw", "screwdriver", 
    "drill", "bit", "plier", "wrench", "skate", "replanting", "soil", "nursery", 
    "pot", "plant", "shovel", "transplant", "rake", "mulch", "water", "spray", 
    "blower", "thread", "needle", "cotton", "bag", "futon", "planet", "yarn", 
    "crochet", "hook", "monopoly", "community", "chest", "property", "card", "dollar", 
    "player", "piece", "dice", "house", "hotel", "poker", "coin", "flute", 
    "drum", "drumstick", "ukelele", "keyboard", "triangle", "strike", "otamatone", 
    "zither", "boxing", "stand", "base", "pump", "hand", "wrap", "dog", "vacuum",
     "toy", "showerhead", "hairdryer", "razor", "shampoo", "ear", "umbrella",
    "solution", "hat", "toothbrush", "toothpaste", "comb", "rag", "velcro", 
    "collar", "mannequin", "toner", "pad", "serum", "dropper", "moisturizer", 
    "foundation", "makeup", "sponge", "powder", "brush", "eyeshadow", "eyeliner", 
    "mascara", "fake", "eyelash", "tweezers", "spoolie", "pencil", "lipstick", 
    "tissue", "buffer", "polish", "earring", "necklace", "bracelet", "ring", 
    "remover", "cleanser", "dispenser", "gym", "bag", "shoe", "cloth", "cleaner",
    "t-shirt", "stain", "removal", "iron", "pants", "sock", "tie", "wipe",
    "lotion", "kit", "ice", "gauze", "adhesive", "bandage", "pbt", "pack",
)

Act_list = (
    "abduct", "absorb", "access", "accord", "add", "adjust", "aerate", "aim",
    "align", "allow", "alter", "alternate", "apply", "arc", "arrange", "assemble",
    "assess", "attach", "avoid", "bake", "balance", "bathe", "beat", "bend",
    "bind", "blend", "block", "blow", "bookmarke", "box", "break", "brighten",
    "bring", "brush", "buckle", "build", "bump", "capture", "carry", "carve",
    "cast", "catch", "center", "change", "charge", "chase", "check", "choose",
    "chop", "circulate", "clamp", "clap", "clasp", "clean", "cleanse", "clench",
    "click", "climb", "clip", "close", "coat", "collect", "comb", "combine",
    "compare", "complete", "compose", "compress", "connect", "consume", "contain", "container",
    "continue", "contrast", "control", "convey", "cook", "cool", "coordinate", "copy",
    "correct", "count", "cover", "crack", "craft", "create", "crimp", "crochet",
    "cross", "crumple", "crush", "cultivate", "cut", "dab", "damage", "darken",
    "deal", "decorate", "decrease", "deflate", "deliver", "depend", "designate", "desire",
    "detach", "develop", "dice", "dig", "dip", "disassemble", "discard", "disconnect",
    "dispense", "display", "dispose", "distribute", "divide", "dodge", "drag", "drain",
    "draw", "dress", "drill", "drink", "drive", "drizzle", "drop", "dry",
    "dump", "dust", "ease", "eat", "edit", "eject", "eliminate", "empty",
    "enclose", "encourage", "engage", "enhance", "enjoy", "ensure", "enter", "erase",
    "establish", "examine", "expand", "expel", "extend", "extract", "face", "fall",
    "fasten", "feed", "feel", "fetch", "fill", "film", "find", "finish",
    "fit", "fix", "flap", "flatten", "fle", "flex", "flick", "flip",
    "float", "flour", "flow", "focus", "fold", "follow", "form", "fry",
    "gather", "gesticulate", "gesture", "get", "give", "gleam", "gluten", "go",
    "grab", "grasp", "grate", "grind", "grip", "groom", "guide", "hammer",
    "handle", "hang", "have", "help", "hide", "hit", "hold", "honk",
    "hook", "hose", "hover", "identify", "illuminate", "illustrate", "improve", "incorporate",
    "increase", "indicate", "inflate", "initiate", "insert", "inspect", "interlock", "intersect",
    "intertwine", "iron", "keep", "knead", "knit", "knock", "knot", "latch",
    "lather", "lay", "lean", "leave", "lift", "light", "lock", "log",
    "look", "loop", "loosen", "love", "lower", "lubricate", "maintain", "make",
    "mark", "massage", "measure", "melt", "mimic", "mince", "miss", "mix",
    "moisturize", "mop", "mount", "move", "mulch", "navigate", "need", "note",
    "nut", "offset", "open", "pack", "page", "paint", "pass", "passcode",
    "paste", "pat", "pause", "pay", "peel", "penetrate", "perform", "pet",
    "pick", "pin", "pinch", "pipe", "place", "plant", "play", "pluck",
    "plug", "position", "pot", "pound", "pour", "practice", "praise", "predetermine",
    "prepare", "press", "pressurize", "pretend", "produce", "protect", "provide", "pull",
    "pump", "punch", "puncture", "push", "put", "raise", "reach", "read",
    "rearrange", "reassemble", "reattach", "receive", "recommend", "record", "reduce", "reinsert",
    "reinserte", "relax", "release", "remain", "remove", "remover", "rename", "reopen",
    "reorder", "repacke", "repair", "repeat", "replant", "reposition", "reseal", "resonate",
    "rest", "retain", "retract", "retrieve", "return", "reveal", "reverse", "rinse",
    "roll", "rotate", "rub", "ruffle", "run", "salad", "sand", "sanitize",
    "save", "saw", "scale", "scan", "scatter", "scoop", "score", "scramble",
    "scrape", "scratch", "screw", "scroll", "scrub", "scrunch", "seal", "search",
    "seat", "secure", "see", "select", "send", "separate", "set", "shake",
    "shape", "sharpen", "shave", "shear", "shield", "shine", "shoot", "shorten",
    "show", "shred", "shrink", "shuffle", "shut", "side", "sieve", "sift",
    "sign", "silence", "sip", "slap", "slash", "sleep", "slice", "slide",
    "slit", "smooth", "snap", "soak", "sort", "spear", "spill", "spin",
    "split", "spoon", "spray", "spread", "sprinkle", "squash", "squeeze", "stabilize",
    "stack", "stamp", "staple", "start", "step", "stick", "stir", "stitch",
    "stop", "store", "straighten", "stretch", "strike", "strip", "stroke", "strum",
    "submerge", "sugar", "support", "suspend", "swab", "swap", "sway", "sweep",
    "swerve", "swing", "swinge", "swipe", "swirl", "switch", "swivel", "tail",
    "take", "tangle", "tap", "tape", "tear", "tend", "test", "texte",
    "thread", "throw", "thrust", "tidy", "tie", "tighten", "tilt", "tip",
    "toast", "toss", "touch", "track", "transfer", "trap", "trash", "trick",
    "trigger", "trill", "trim", "tuck", "tune", "turn", "twist", "type",
    "unbuckle", "unbutton", "uncap", "unclampe", "unclasp", "unclench", "unclenche", "unclip",
    "unclippe", "uncover", "unfasten", "unfold", "unhook", "unknotte", "unlatch", "unlatche",
    "unlock", "unpack", "unplante", "unplug", "unroll", "unscrew", "unseal", "unsheathe",
    "untangle", "unthread", "untie", "unwiden", "unwind", "unwrap", "unwrenche", "uproot",
    "upturn", "use", "valve", "vary", "view", "waggle", "wait", "warm",
    "wash", "water", "wave", "wear", "weave", "wedge", "widen", "wiggle",
    "wind", "wing", "wipe", "withdraw", "wobble", "work", "wrap", "wrench",
    "wring", "write", "yarn", "zip", "zoom"
)


Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        self.word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[self.word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split('/')
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec['unk']
            pos_vec = self._get_pos_ohot('OTHER')
        return word_vec, pos_vec
