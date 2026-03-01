# Presentation Script (~2 min)

GenAI is not good at song lyrics. But why? And can we do better?

LLMs get the basics wrong. LLMs often write too much and use a really wide vocabulary, like they are trying to show off They are obsessed with rhyming, they hate slang and they hate repeating themselves. 
the vibe is just off.

So, we're gonna try and fix it.

I grabbed a dataset of 5 million song lyrics to get a baseline.

I made a self-improving prompt engineering loop using Mistral 7b, weights and biases and claude code. 
The tricky part was the evals. What are we basing the scoring on? The whole point is the LLMs are not well suited for this, so LLM-as-judge is a problem. Nonetheless, I picked out authenticity, repetition, unique words, and rhyme as areas the genAI lyrics differed most from real ones. 
This was.. ok. We got some better prompts. 

But I think we could do better.

So - fine tuning time. I fine-tuned Mistral7B w/ Lora, on HuggingFace jobs which made things pretty simple, with Weights & Biases logging everything.
I used 200k songs from our real lyrics dataset.

Result! The outputs to me felt far closer to natural song lyrics.

I was quite worried about my own bias though, so I built a website where people can blind-judge lyrics side by side — you see two versions, pick which one feels more like a real song.
My fine-tuned versions won out.. although it wasn't overwhelming. Looking at lyrics in isolation is a bit odd and some folks preferred the creativity of the base model.

