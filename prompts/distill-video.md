You are distilling one YouTube video into study notes. The transcript below came from automatic speech recognition of {language} speech, often mixed with English (e.g. {language} with English finance terms). Expect recognition errors: misheard English loanwords written in {language} script, missing punctuation, wrong word boundaries. Infer the intended meaning from context and the glossary, but never invent content that is not supported by the transcript.

Video: {title}
Channel: {channel}
Uploaded: {upload_date}
URL: {url}
{chapters}
Write the notes in English. When the speaker uses a notable {language} term or phrase, keep it in brackets after the English, written in {language} script as the speaker said it, e.g. "risk appetite (<the {language} phrase>)". Never use terms from any other language. Cite the transcript timestamp for every bullet as [mm:ss] so each point can be checked against the video.

Output only Markdown, with exactly these sections (omit a section only if the video truly has nothing for it):

## TL;DR
2–4 sentences: what this video teaches and the speaker's main position.

## Key concepts
Each concept: a bolded name, a plain-English definition as the speaker explains it, and any example they use.

## Rules, frameworks and checklists
Concrete decision rules, step-by-step methods, thresholds and numbers the speaker gives.

## Worked examples
Any calculations or case studies, with the numbers.

## Claims and opinions
Market views, predictions, product or stock opinions, labelled as the speaker's opinion. Flag anything time-sensitive (tax rules, rates, schemes) with ⏳.

## Action items
What the speaker tells the viewer to do.

## Transcript quality
One line: how reliable the transcript seemed, and any passages you could not interpret.

Glossary of domain terms (correct spellings; the transcript may garble them):
{glossary}

Transcript:
{transcript}
