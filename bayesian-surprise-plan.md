Bayesian surprise plan
======================

This plan is designed to measure Bayesian surprise (KL divergence over belief distribution) and compare it to two simpler representations of surprise or novelty.

Online reviews give us evidence about passages in a book that human readers believed were likely to be unexpected. It's not exhaustive, but those reviews presumably flag at least some of the most intensely surprising twists in the plot. We can distinguish 1000-word passages that contain one of those revelations and 1000-word passages that don't. The next passage could optionally be included, on the hypothesis that every discovery has a ripple effect.

We track two latent variables throughout the book: 1) the main conflict and 2) the central mystery (what the reader expects to be revealed by the end of the book). At a future time we could add latent variables for characters (descriptions and or motives).

For each latent, at each time step, we will measure KL divergence from the distribution in each of the last three time steps. We could later expand that to five or ten time steps, if desired.

But right now, this will give us Bayesian surprise (change in beliefs) across a 4,000-word span. We are explicitly not attempting to track surprise across the whole narrative at once; our theory is that readers' beliefs change enough across a book that the last chapter or two are most relevant for surprise.

This could be a wrong assumption, but it saves us from the otherwise quite difficult task of producing a coherent summary, at each time step, that represents a reader's integrated understanding of the whole previous plot.

Instead we give one-sentence summaries of the last seven time steps for local context.

To create some coherence across time, we also present summary descriptions of four central characters.

We're going to run the experiment first with Mixtral 8 x 7B (4-bit quantized) which was released in Dec 2023, and then later with a recent strong model (likely Gemma 4 Dense 31B, quantized to 8bit). We'll use instruction-tuned versions of the models to generate beliefs and summaries, but use base versions to assess the probability of beliefs.

My strong suspicion is that with models of this size, and medium-obscure books, it actually doesn't matter whether they were released before or after the novels. I don't think a 31B model memorizes plot twists in every random thriller. But we cannot ask scholars to trust us on that; if Gemma is as surprised as Mixtral, and in the same places, we'll have evidence.

Our baselines for contrast are:

 a) Embedding novelty. Take embeddings of the last three chunks and contrast embeddings of the new chunk. Take cosine distance. This should be a very bad measure of surprise.

 b) Improbability of the next summary. Give the model everything it's normally given at time T, ask what will happen next? and measure the log-likelihood of the empirically-observed summary at T+1 as a completion of the question.

 c) Refinement of improbability. Generate 7 actual predictions of what happens next, based on knowledge at T, and add the actual T+1 summary to create 8 options. Measure the completion logits for each of these options individually to get the perceived probability of the actual "what happens next" as one element in the distribution.

One reason to believe Bayesian surprise will outperform these baselines: it focuses on beliefs about durable latent variables (conflicts, mystery, characters) rather than literally what happens next. So it will be more resistant to flashbacks and nonlinear multi-plot narration.

preparation
-----------

For each novel, before doing analysis, we optionally create a character file identifying up to seven central characters, their aliases, and the chunk-index range within which each character's description may be included in the prompt. We also write a one-sentence premise describing the setting/narrator that is permanently true throughout the story.

The format is a JSON object (not a list) with a top-level "premise" key and then up to seven entries keyed by the character's canonical display name:

      {
        "premise": "The novel is set in contemporary Chicago and is narrated in the first person by Meghan Michaels.",
        "Meghan Michaels": {"aliases": ["Meghan", "Meghan Michaels", "Ms. Michaels"], "range": [0, 99]},
        "Caitlin Beckett":  {"aliases": ["Caitlin", "Caitlin Beckett", "Ms. Beckett"],  "range": [0, 99]},
        ...
      }

The inclusive chunk range [start, end] lets us handle the case where two characters are later revealed to be the same person: create two separate entries with early ranges (when each is known as a distinct individual) and then a fused entry (e.g. "Caitlin / Jane") with a later range. Only the entries whose range covers the current step T are eligible to appear.

A character "counts as mentioned" when any of its aliases matches the text on a word boundary (so the short alias "Nat" does not false-match "Nathan"). This check applies to the new chunk and to the last seven step-summaries.

This work is done externally to the main script by a human, an LLM, or both. The file lives in characters/, following a naming convention such that chunkedtexts/ShesNotSorryMaryKubica.json is paired with characters/ShesNotSorryMaryKubica_characters.json.

We also divide the model into chunks as close as possible to 1000 words while breaking at paragraphs. This produces a Text object, where e.g. Text_0 is the first chunk.

Then we step through the novel chunk by chunk.

what we show the model at each time step
----------------------------------------

This should be encapulated as a function that accepts the Text object and Summary object, along with a a time step T and a variable specifying anything to_be_hidden. It returns a prompt.

At every step, if a premise is defined it appears first: "Premise:\n<premise>".

At step 0 the prompt is the premise (if any) followed by Text_0.

At each step after 0:

We identify the subset of up to seven characters whose chunk range covers T and who are mentioned (by any alias, word-boundary) in the new passage or in the summaries of the last seven steps. For each, we present their canonical name and the most recent one-sentence summary description of the character, unless that character's canonical name is in to_be_hidden.

We also present one-sentence summaries of the last seven time steps.

So overall the prompt is: Premise → Characters (up to seven descriptions) → Recent events (up to seven summaries) → Text_t.

what we ask the model at each time step
---------------------------------------

We elicit several things at each time step, using an instruction-tuned model.

#### belief probabilities

At each step, we ask the latent question for each of two latents: "What is a central conflict in this story?" and "What is something the reader expects to discover by the end of the story?" 

[In the future, if we start measuring character surprise: If character C is mentioned, we ask for beliefs about character C, removing the summary description of C when we ask this.]

In each of these cases we oversample beliefs, to maximize diversity, using this technique: 

We ask the model to generate five distinct answers (distinct ways of expressing the conflict, distinct things the reader expects to learn, &c). E.g.

    What is the central conflict in this story?
    Provide five distinct answers, each a short sentence on a numbered new line, and beginning with the same phrase: 

    1. In this story, the central conflict is

We repeat this 4 times so we have 20 possible options. 

Then we measure the log-likelihood of each option separately, using the same prompt and checking the probability of everything after "In this story, the central conflict is" or "By the end of this story, the reader will discover." Note that in this situation (exactly the same longish prompt, scoring 20 possible continuations) we likely want to use a base model (which gives more exact likelihoods for continuation). Also, if we use KV caching to store the prompt we can greatly speed up inference.

From those 20 options, we use Maximal Marginal Relevance (Carbonell and Goldstein 1998) to select the 8 options that maximize a weighted combination of relevance to the query (likelihood) and dissimilarity from already-selected items.

Once we have 8 diverse alternatives, we save them. We also already have the logits of each alternative.

This gives us a probability distribution over 2 latent variables for time step T. [In the future, we could have 2-8 latents, depending on how many central characters are mentioned.]

#### elicit new summaries that will be presented as context

We also ask the (IT) model for a one-sentence summary of the new text chunk, which will be presented as background in the next time step. And, if character C was mentioned, we present both the first summary description of C and the most recent summary description of C, and ask for a new summary description of C — the persistence of previous descriptions here is the one way we attempt to create continuity across the whole narrative.

#### elicit what happens next beliefs

We ask the model:

    Predict what happens next in the form of a short present-tense sentence. E.g. "Red Riding Hood arrives at her grandmother's house." Provide five distinct predictions, each numbered and on a new line.

    1.

Ask that question twice, asking for a list of five in each case, to get 10 options. Use MMR to reduce to 7. Evaluate their likelihood after the prompt for T plus "What happens next:"

data structure saved at each time step
----------------------------------------

In the Summary data structure:
New summary_t
Character_summaries keyed by canonical name, for up to 7 characters  # if a new one is not generated use the last

In the Belief data structure:

L_conflict_beliefs = [list of eight strings]
L_conflict_logits = []
L_mystery_beliefs = [list of eight strings]
L_mystery_logits = []
L_next_beliefs = [list of seven strings]
L_next_logits = []

Measuring Bayesian surprise
---------------------------

Once all the data is generated for each time step, we can step again through the structure to measure Bayesian surprise and the improbability of what-happens-next at each step.

#### Bayesian surprise

Measure KL divergence from the last three time steps. We do this by 

    a) constructing the prompt for time T, 
    b) stepping through T-3, T-2, T1, in each case recalling 8 beliefs that were *generated at the time* for each of two latent plot variables. 
    c) Measure the log-likelihood of each belief when it follows the prompt for time T.

In the future: we'll do this across the last five time steps.

We will also try a pooled measure of surprise where we *pool* the expressed beliefs about latent L at T and T+n, and compute a probability distribution over this pooled support both at T and T+n, then compute the KL divergence between those distributions. The support is the same in both cases. This departs from the normal definition of Bayesian surprise, which involves only a change in distribution over fixed support. But intuitively, it seems to make sense to consider both the decreased probability of past beliefs and the increased probability of future ones. It's still asymmetric, because we use KL divergence from T => T+n, not Jensen-Shannon.

#### what happens next

Evaluate likelihood of the summary_t+1 after the prompt of T. Add it to the 7 what-happens-next beliefs at T and take softmax to get a probability distribution. Save the probability of summary_t+1.

number of total model calls per step
------------------------------------

At each step we have 1 call to elicit the new summary, and up to 7 calls for the new character summaries = 8.

Then 4 calls to elicit 5 beliefs each for each latent, or 2 x 4 = 8.

Then, for each latent, get the log-likelihood of 20 beliefs = 2 x 20 = 40.

To compute surprise, we need to evaluate the new likelihoods of 8 beliefs for 2 latents for each of the past three time steps. 8 x 2 x 3 = 48. Exactly the same prompt each time so this can be greatly sped up with KV caching.

If we do pooled surprise, we also evaluate the likelihoods of our current 8 beliefs for 2 latents in each of the past three steps. 8 x 2 x 3 = 48. Here there are three different prompts, but it's still possible to reuse a cache 16 times.

So for Bayesian surprise, up to 152 calls per time step (149 + 3 extra for the larger character cap).

For naive improbability, two calls to elicit 5 "what happens next?" Plus 10 to evaluate likelihoods. Use MMR to reduce to 7. Plus 1 for the actual likelihood of T+1. 13 calls per time step.

Total of 162 per step x let's say 100 steps, or ~16,200 per novel. Very achievable with caching.

