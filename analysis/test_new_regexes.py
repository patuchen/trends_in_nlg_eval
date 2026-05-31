import re
import sys

_stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(_stdout_reconfigure):
    _stdout_reconfigure(encoding="utf-8")

METHOD_RES_NEW = {
    "likert": re.compile(r'\blikert\b|\d[\s-]?point\s+(scale|rating|likert)|scale\s+of\s+\d[^\.]*?to\s+\d|\d[\s-]to[\s-]\d\s+scale', re.IGNORECASE),
    "pairwise": re.compile(r'pairwise\s+(comparison|preference|evaluat|judg)|side[\s-]by[\s-]side|forced[\s-]choice|\bA/B\s+test(ing)?|preference\s+(study|test|judg)', re.IGNORECASE),
    "categories": re.compile(r'binary\s+(annotation|evaluat|judg|rating|choice)|yes[\s/]no\s+(evaluat|judg|rating)|acceptability\s+judg|fluent\s+or\s+not|\bcategories:\s+[A-Z]', re.IGNORECASE),
    "span": re.compile(r'span\s+annotation|error\s+(span|mark|annot)|span[\s-]?(level|based)\s+(evaluat|annot)|error\s+(highlight|tag)|\bMQM\b|Error\s+Span\s+Annotation', re.IGNORECASE),
    "best_worst_scaling": re.compile(r'best[\s-]worst\s+scal|\bBWS\b', re.IGNORECASE),
    "ranking": re.compile(r'\branking\s+(evaluat|annot|study)|human\s+ranking|rank\s+(order\s+)?(the\s+output|outputs|the\s+response|the\s+translation|the\s+summar)|relative\s+ranking\s+of', re.IGNORECASE),
    "direct_assessment": re.compile(r'direct\s+assessment|\bDA\b\s+(evaluat|score|protocol|rating)|adequacy[\s/]fluency\s+rating', re.IGNORECASE),
    "post_editing": re.compile(r'post[\s-]?edit(ing|ed)?|postedit(ing|ed)?|human[\s-]?post[\s-]?edit(ing|ed)?|\bHTER\b', re.IGNORECASE),
}

current_method = None
samples = []

with open('regex_samples.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith('=== Sampling'):
            if current_method and samples:
                # evaluate the new regex on the old samples
                regex = METHOD_RES_NEW.get(current_method)
                if regex:
                    matches = [s for s in samples if regex.search(s)]
                    print(f"\nMethod: {current_method}")
                    print(f"Old matches: {len(samples)}")
                    print(f"New matches: {len(matches)}")
                    print(f"Discarded: {len(samples) - len(matches)}")
                    if current_method in ['likert', 'pairwise', 'categories', 'span', 'ranking']:
                        print("First 5 DISCARDED (should be overfirings):")
                        discarded = [s for s in samples if not regex.search(s)]
                        for d in discarded[:5]:
                            print("  - " + d)
            current_method = line.split('Sampling ')[1].split(' ===')[0]
            samples = []
        elif re.match(r'^\d+\.\s+\[', line):
            samples.append(line)

# process the last one
if current_method and samples:
    regex = METHOD_RES_NEW.get(current_method)
    if regex:
        matches = [s for s in samples if regex.search(s)]
        print(f"\nMethod: {current_method}")
        print(f"Old matches: {len(samples)}")
        print(f"New matches: {len(matches)}")
        print(f"Discarded: {len(samples) - len(matches)}")
        if current_method in ['likert', 'pairwise', 'categories', 'span', 'ranking']:
            print("First 5 DISCARDED (should be overfirings):")
            discarded = [s for s in samples if not regex.search(s)]
            for d in discarded[:5]:
                print("  - " + d)

