**Contents** [1. How to answer](#how) [2. Core concepts](#core) [3. Framing and data](#framing) [4. Choosing a model](#models) [5. Training decisions](#training) [6. Evaluation decisions](#eval) [7. Trade-offs](#tradeoffs) [8. Diagnosing problems](#debug) [9. Worked cases](#cases) [10. Deployment](#prod) [11. Rapid-fire Q and A](#rapid) [12. Common traps](#traps) [13. Cheat sheet](#cheat) [14. Practice plan](#practice)

# ML Fundamentals: Interview Preparation

Core concepts, model design, training, evaluation, and trade-offs, written for an interview where the real test is explaining *why* you chose something, not only *what* you chose.

**The one idea behind this guide.** Interviewers rarely want a correct textbook answer. They want to see a decision under constraints: you name the options, say what would change your mind, pick one, and state the cost of your choice. Every section below is built around that. Green “Because” lines are the reasoning to say out loud.

## 1. How to answer a decision question

Use this skeleton for any “which model / metric / approach would you use?” question. It keeps you from jumping to a favorite tool.

1. **Clarify the goal and the cost of errors.** What decision does the prediction drive? Which mistake is worse?
2. **State the constraints.** Data size, label quality, latency, interpretability, retraining cadence, team skills.
3. **Propose a baseline first.** Something simple you can ship and measure against.
4. **Name two or three candidate approaches** and what each assumes.
5. **Choose, and say why** in terms of the constraints you just stated.
6. **Name the main risk** of your choice and how you would detect it.
7. **Say what would make you switch.** This is the strongest signal of real judgment.

### Phrases that signal good reasoning

- “It depends on X. If X is large I would…, otherwise I would…”
- “I would start with a baseline so we know what the complexity buys us.”
- “The risk with that is… and I would check it by…”
- “I am optimizing for A at the expense of B, because in this setting A matters more.”

### Phrases to avoid

- “I would just use a neural network / XGBoost.” (No reasoning given.)
- “Always” or “never” about techniques. Nearly everything has a regime where it flips.
- Reciting definitions without connecting them to a choice.

Worked micro-example: “Why would you start with logistic regression?”

**Answer**It gives me a fast, well-calibrated, interpretable baseline.

**Because**If a complex model only beats it by a point of AUC, the added maintenance, latency, and opacity may not be worth it. If the gap is large, I have evidence the extra complexity earns its keep, and the baseline coefficients also help me sanity-check the data.

**Switch if**Strong feature interactions or non-linearities show up in error analysis, and I have enough data to support a more flexible model.

## 2. Core concepts you must be able to explain

### Generalization and the bias-variance trade-off

Expected test error decomposes (for squared loss) into **bias²** (error from wrong assumptions, underfitting), **variance** (sensitivity to the particular training sample, overfitting), and **irreducible noise**. More model capacity lowers bias and raises variance; more data lowers variance; regularization trades a little bias for less variance.

Training error is low but validation error is high. What do you do?

**Answer**Treat it as a variance problem: get more data or augment it, regularize more, simplify the model, use early stopping, or ensemble.

**Because**A large train-validation gap means the model has memorized sample-specific noise. Making the model bigger would widen the gap. If instead both errors are high, it is a bias problem and I would add capacity or better features.

### Loss, likelihood, and why the usual losses are what they are

- **Squared error** is maximum likelihood under Gaussian noise. It is sensitive to outliers because errors are squared.
- **Absolute error** corresponds to Laplace noise; its minimizer is the conditional median, so it is robust to outliers.
- **Cross-entropy / log loss** is maximum likelihood for a Bernoulli or categorical output. It penalizes confident wrong predictions heavily and gives well-behaved gradients with a sigmoid or softmax.
- **Hinge loss** (SVM) only cares about points inside or on the wrong side of the margin.

Why use cross-entropy instead of MSE for classification?

**Answer**Cross-entropy matches the probabilistic model and trains better.

**Because**With a sigmoid output, MSE gradients shrink when the unit saturates, even if the prediction is badly wrong, so learning stalls. Cross-entropy’s gradient with respect to the logit is simply (prediction minus label), so a confidently wrong prediction produces a large corrective gradient.

### Regularization as a prior

- **L2 (ridge / weight decay)** equals a Gaussian prior on weights. It shrinks weights smoothly and handles correlated features by spreading weight across them.
- **L1 (lasso)** equals a Laplace prior. Its constraint region has corners on the axes, so solutions often land exactly at zero, giving sparsity and built-in feature selection.
- **Elastic net** mixes both: sparsity from L1, stability under correlated features from L2.

L1 or L2?

**Answer**L2 by default; L1 when I believe only a few features matter or I need a sparse, interpretable model.

**Because**L1 picks one feature arbitrarily from a correlated group, which makes it unstable. L2 keeps all of them with shared weight. If I want both sparsity and stability, elastic net.

### Optimization basics

- **Gradient descent** follows the negative gradient. **SGD / mini-batch** estimates it from a subset: cheaper per step, noisier, and the noise can help escape sharp minima.
- **Momentum** smooths direction; **Adam** adapts per-parameter step sizes using running estimates of the gradient’s first and second moments.
- **Convex** problems (linear and logistic regression, SVM) have a single global minimum. Neural networks are non-convex; in practice good local minima are usually fine.

### Probabilistic ideas worth knowing cold

- **MLE vs MAP:** MAP is MLE plus a prior; regularization is the prior.
- **Discriminative vs generative:** discriminative models learn p(y|x) directly; generative models learn p(x,y). Generative can be better with little data or missing features; discriminative usually wins on accuracy with ample data.
- **Calibration:** whether a predicted 0.8 really happens about 80% of the time. Ranking quality (AUC) and calibration are separate properties.

**Key points**

- Every loss encodes an assumption about noise; choosing one is choosing an assumption.
- Regularization is a deliberate trade of bias for variance.
- Diagnose with the train-validation gap before choosing a fix.

## 3. Problem framing and data decisions

Most real-world failures happen here, and interviewers know it. Strong candidates talk about framing before models.

Before touching a model, what do you establish?

**Answer**The decision the model supports, the metric that reflects its business cost, a baseline, and how the data will be split.

**Because**A model optimized for the wrong metric, or evaluated on a leaky split, produces a confident number that is meaningless. Fixing that later costs far more than a day spent up front.

### Splitting strategy: match deployment

| Situation | Split | Why |
| --- | --- | --- |
| Independent, identically distributed rows | Random (stratified for classification) | Matches the assumption; stratifying keeps rare-class proportions stable. |
| Time-dependent data (forecasting, churn, fraud) | Train on the past, validate on the future (rolling or expanding window) | Random splits leak future information and inflate scores; production only ever predicts forward. |
| Multiple rows per user, patient, or device | Group split by entity | Otherwise the model memorizes entities and looks great on “new” rows that are not new. |
| Small dataset | k-fold cross-validation, repeated if needed | A single holdout is high-variance; CV uses data efficiently and gives an error bar. |
| Hyperparameter tuning plus honest estimate | Train / validation / test, or nested CV | Tuning on the test set optimistically biases it; the test set must be touched once. |

### Data leakage: the most common hidden failure

- **Target leakage:** a feature that encodes the outcome or is only known after it (for example “refund issued” when predicting fraud).
- **Train-test contamination:** fitting scalers, imputers, encoders, or feature selection on all data before splitting. Fit on training folds only, ideally inside a pipeline.
- **Duplicates and near-duplicates** across splits.

**Red flag you can volunteer:** “If a model’s accuracy looks suspiciously good, my first suspicion is leakage, and I check feature importances for something that should not be predictive.”

### Class imbalance

Fraud occurs in 0.2% of transactions. How do you approach it?

**Answer**Pick imbalance-aware metrics (PR-AUC, recall at a fixed precision, cost-weighted loss), keep the evaluation set at the true class ratio, and start with class weights before resampling. Tune the decision threshold to the cost of errors.

**Because**Accuracy is 99.8% for a model that never flags fraud. Resampling the evaluation set would hide how the model behaves in production. Class weights change the loss without altering the data distribution; oversampling (including SMOTE) can add noise or overfit duplicates, and under-sampling discards information. Thresholding is cheaper and more controllable than rebalancing.

**Caveat**Reweighting or resampling distorts predicted probabilities. If calibrated probabilities matter, recalibrate afterward on data with the true ratio.

### Missing values, outliers, and features

- **Missing data:** add an “is missing” indicator, since missingness is often informative. Impute with median or a model; many tree libraries handle missing values natively. Dropping rows is acceptable only if missingness is rare and unrelated to the target.
- **Outliers:** first ask whether they are errors or real rare events. Robust losses (Huber, MAE), winsorizing, or log transforms are options; deleting real extremes can remove exactly the cases you care about.
- **Scaling:** required for distance-based and gradient-trained models (kNN, SVM, regularized linear models, neural nets); unnecessary for tree-based models.
- **Categorical encoding:** one-hot for low cardinality; target encoding (with out-of-fold fitting to prevent leakage) or learned embeddings for high cardinality.

## 4. Choosing a model

The correct answer is almost always “it depends on the data and constraints.” This table gives you the dependencies.

| Model family | Reach for it when | Why | Watch out for |
| --- | --- | --- | --- |
| Linear / logistic regression | Baseline; need interpretability, speed, calibrated probabilities; small data; many features | Convex, stable, coefficients explain themselves | Cannot capture interactions or non-linearity without manual features |
| Decision tree | Need human-readable rules | Handles mixed types, no scaling | High variance; one tree overfits easily |
| Random forest | Strong default with little tuning; noisy tabular data | Averaging many decorrelated trees cuts variance | Large and slower at inference; cannot extrapolate |
| Gradient boosting (XGBoost, LightGBM, CatBoost) | Tabular data where accuracy matters | Sequentially fits residuals, reducing bias; often best on structured data | Needs tuning; overfits if too deep or too many rounds; cannot extrapolate |
| SVM | Small to medium data, high-dimensional, clear margin | Max-margin generalization; kernels for non-linearity | Poor scaling past \~100k samples; no native probabilities |
| k-nearest neighbors | Quick baseline; local structure; few features | No training, assumption-light | Slow at inference; breaks in high dimensions; needs scaling |
| Naive Bayes | Very small data, text baselines, need speed | Few parameters, low variance | Independence assumption; poorly calibrated |
| Neural networks | Images, text, audio, sequences; very large data; learned representations | Learn features end to end | Data-hungry, costly, harder to debug and explain; usually not best on small tabular data |
| Clustering / dimensionality reduction | No labels; exploration; preprocessing | Reveal structure | Results depend heavily on scaling and the chosen k or method; validate with downstream use |

### Bagging vs boosting

Random forest or gradient boosting?

**Answer**Random forest if I want a robust model quickly with minimal tuning or my labels are noisy; boosting if I have time to tune and want the best accuracy.

**Because**Bagging reduces *variance* by averaging independent high-variance trees, so it is hard to break. Boosting reduces *bias* by adding trees that correct earlier errors, so it reaches higher accuracy but will also fit noisy labels if unchecked, which is why learning rate, depth, and early stopping matter.

### Tabular data and deep learning

Would you use a neural network on a 50,000-row tabular dataset?

**Answer**I would start with gradient boosting and only try a neural net if I need joint training with other modalities, embeddings for high-cardinality features, or online updating.

**Because**On moderate tabular data, boosted trees typically match or beat neural nets with far less tuning, are faster to train, and handle heterogeneous features and missing values natively. I would revisit if the data grows by orders of magnitude or the problem has structure that trees cannot exploit.

### Complexity is a cost

When two models perform within noise of each other, choose the simpler. Say this explicitly: “A 0.3% gain within the cross-validation standard error is not a real gain, and I would take the cheaper, more interpretable model.”

## 5. Training decisions

### Optimizer and learning rate

Adam or SGD with momentum?

**Answer**Adam (or AdamW) to get a good result quickly with little tuning; SGD with momentum when I have time to tune and want the best generalization, as in many vision settings.

**Because**Adam’s per-parameter scaling makes it robust to poorly scaled gradients and sparse features, so it converges fast out of the box. Well-tuned SGD often finds flatter minima that generalize slightly better, at the cost of more learning-rate and schedule tuning. For Adam, decoupled weight decay (AdamW) regularizes correctly, which plain L2 inside Adam does not.

How do you pick a learning rate?

**Answer**Run a learning-rate range test, pick a value just below where loss starts to diverge, and use warmup plus a decay schedule.

**Because**It is the most influential hyperparameter. Too high diverges or oscillates; too low is slow and may stall in poor regions. Warmup avoids early instability when estimates are noisy; decay lets the model settle into a minimum.

### Batch size

- **Small batches:** noisier gradients, a mild regularizing effect, lower memory.
- **Large batches:** better hardware utilization and stable gradients, but can generalize worse unless the learning rate is scaled and warmed up.

**Because** noise acts as implicit regularization, but throughput matters at scale. The practical answer: use the largest batch that fits comfortably, scale the learning rate with it, and verify validation performance.

### Controlling overfitting: a menu with reasons

| Technique | What it does | Use when |
| --- | --- | --- |
| More data / augmentation | Reduces variance at the source | Always the first thing to consider; augmentations must preserve the label |
| Weight decay (L2) | Shrinks weights, smoother functions | Default for nearly everything |
| Early stopping | Stops when validation loss turns up | Cheap and effective; acts like implicit regularization |
| Dropout | Randomly drops units, prevents co-adaptation | Large fully connected layers; less useful with batch norm or in some architectures |
| Simpler model / fewer features | Lower capacity | Small data |
| Ensembling | Averages out variance | When accuracy matters and inference cost allows |

### Normalization and initialization

- **Input standardization** makes the loss surface better conditioned so gradient methods converge faster.
- **Batch norm** stabilizes activations and allows higher learning rates; **layer norm** is preferred when batch statistics are unreliable (small batches, sequence models, transformers).
- **Initialization** (Xavier for tanh/sigmoid, He for ReLU) keeps activation variance stable across layers so signals and gradients neither vanish nor explode.
- **Vanishing gradients:** mitigated by ReLU-family activations, residual connections, normalization, and careful initialization; **exploding gradients** by gradient clipping.

### Hyperparameter search

Grid, random, or Bayesian search?

**Answer**Random search by default; Bayesian optimization when each trial is expensive.

**Because**Usually only a few hyperparameters matter a lot, and a grid wastes trials repeating the same values of the important ones, while random search tries more distinct values of each. Bayesian methods use past results to pick promising trials, which pays off when evaluations are costly.

## 6. Evaluation decisions

### Choosing a metric: start from the cost of mistakes

| Metric | Choose it when | Why, and the catch |
| --- | --- | --- |
| Accuracy | Balanced classes, symmetric error costs | Simple, but misleading under imbalance |
| Precision | False positives are expensive (spam filter blocking real mail, manual review capacity) | Of the flagged, how many are right |
| Recall | False negatives are expensive (cancer screening, fraud) | Of the real positives, how many we catch |
| F1 / Fβ | Need one number balancing precision and recall; β>1 favors recall | Ignores true negatives; threshold-dependent |
| ROC-AUC | Comparing ranking quality; roughly balanced classes | Threshold-free, but can look optimistic under heavy imbalance because the false positive rate is diluted by many negatives |
| PR-AUC | Rare positive class | Focuses on the positive class; depends on prevalence, so compare only on the same data |
| Log loss / Brier score | Probabilities themselves are used downstream | Rewards calibration and punishes overconfidence |
| RMSE | Large errors are disproportionately costly | Squares errors; sensitive to outliers |
| MAE | Cost is linear in error; outliers present | Robust; easier to interpret |
| MAPE / sMAPE | Relative error matters | Breaks near zero actuals; asymmetric |
| NDCG, MAP, recall@k | Ranking and recommendation | Reflects the top of the list, where users look |

ROC-AUC or PR-AUC for a rare-event problem?

**Answer**PR-AUC, plus the operating point I would actually deploy at (for example recall at 90% precision).

**Because**With 1,000 negatives per positive, a model can have a tiny false positive *rate* yet still bury reviewers in false alarms. Precision exposes that cost directly. ROC-AUC remains useful for comparing pure ranking quality.

### Thresholds are a business decision

A classifier outputs scores; the threshold turns them into actions. Choose it from the cost matrix: flag when expected cost of ignoring exceeds expected cost of acting. For costs CFN and CFP with calibrated probabilities, the cost-optimal threshold is CFP / (CFP + CFN). **Because** the default 0.5 is only optimal when errors cost the same and classes are balanced, which is rare.

### Calibration

When do you care about calibration, and what do you do about it?

**Answer**When probabilities feed decisions, expected-value calculations, or are shown to people. I check a reliability diagram and Brier score, then apply Platt scaling or isotonic regression on a held-out set.

**Because**Many strong models (boosted trees, SVMs, modern neural nets) rank well but produce over- or under-confident probabilities. Isotonic needs more data than Platt scaling but is more flexible.

### Validation and statistical honesty

- Report **uncertainty**: standard deviation across CV folds or bootstrap confidence intervals. A 0.5% difference with ±1% noise is not a difference.
- **Keep the test set sealed** until the end; each peek leaks information and biases the estimate.
- **Look at slices:** performance by segment (region, device, new vs returning users). A good average can hide a failing subgroup, which also matters for fairness.
- **Offline metrics are proxies.** Confirm with an online experiment (A/B test) where the real outcome can be measured, with a pre-set sample size and guardrail metrics.

## 7. The trade-offs to be fluent in

| Trade-off | How to reason about it |
| --- | --- |
| Bias vs variance | Increase capacity if both train and validation error are high; regularize or add data if only validation error is high. |
| Precision vs recall | Set by the relative cost of false positives and false negatives; move the threshold, not the model, first. |
| Accuracy vs interpretability | In regulated or high-trust settings (credit, health) a slightly weaker explainable model often wins; use post-hoc explanations (SHAP, partial dependence) with care, since they describe the model, not causality. |
| Accuracy vs latency and cost | A 0.5% lift is not worth a 10x inference cost for a real-time path; distill, prune, quantize, or use a cascade (cheap model first, expensive model on uncertain cases). |
| Exploration vs exploitation | Needed when the model’s own decisions determine future data (recommenders, ads); bandits or randomization prevent feedback loops. |
| Data quantity vs quality | Cleaner labels often beat more labels. Noisy labels cap achievable performance and hurt flexible models most. |
| Build vs buy / pretrained vs scratch | Start from a pretrained model and fine-tune when labeled data is scarce; train from scratch when your domain differs strongly and you have lots of data. |
| Fairness vs raw accuracy | Define the fairness criterion with stakeholders, measure it per group, and note that common criteria (equalized odds, demographic parity, calibration across groups) cannot all hold at once when base rates differ. |
| Simplicity vs flexibility | Prefer the simplest model that meets the requirement; complexity adds maintenance and failure modes. |

## 8. Diagnosing problems: symptom to action

Interviewers love “your model is not working, what do you do?” Answer with a diagnosis order, not a list of random tricks.

### The default order

1. Check the **data and labels** (leakage, bugs, label noise, distribution differences).
2. Check the **pipeline** (the same preprocessing at train and serve time).
3. **Overfit a tiny batch.** If the model cannot memorize 20 examples, there is a bug.
4. Compare against the **baseline** and error-analyze by slice.
5. Only then change the **model** or hyperparameters.

| Symptom | Likely cause | What I would try, and why |
| --- | --- | --- |
| Train and validation both poor | Underfitting, weak features, or a bug | Confirm with a tiny-batch overfit; add capacity, better features, or train longer. A bigger model only helps if the signal exists. |
| Train great, validation poor | Overfitting or leakage in the opposite direction (validation differs from train) | Regularize, add data; also check that the split reflects deployment. |
| Validation great, production poor | Leakage, train-serve skew, drift | Audit features available at prediction time; compare feature distributions in production and training. |
| Loss is NaN or explodes | Learning rate too high, bad input values, log of zero | Lower the learning rate, clip gradients, check for NaN inputs, add epsilon to logs. |
| Loss decreases extremely slowly or plateaus | Learning rate too low, vanishing gradients, poor scaling | Run an LR range test, standardize inputs, check initialization and activations. |
| Validation loss rises while accuracy stays flat | Overconfidence, miscalibration | Label smoothing, temperature scaling, more regularization. |
| Great overall metric, bad for one group | Under-represented segment, proxy features | Slice metrics, gather data for the segment, consider group-aware thresholds, review features. |
| Performance degrades over weeks | Data or concept drift | Monitor input and output distributions, schedule retraining, keep a recent-data holdout. |
| Results change wildly between runs | High variance, small data, unseeded randomness | Fix seeds, repeat runs, report mean and spread; use CV or more data. |

## 9. Worked cases: practice thinking aloud

Read each prompt, answer it yourself in two minutes, then compare. Notice the structure: goal, constraints, baseline, choice, because, risk.

Case A. Churn prediction for a subscription product

**Framing.** The prediction drives a retention offer. A missed churner costs the customer’s lifetime value; a wasted offer costs the discount. So I would rank customers by churn probability times value and size the campaign by budget, which means I need *calibrated* probabilities and a ranking metric (recall or lift at the top k%), not accuracy.

**Split.** Time-based: train on earlier cohorts, validate on later ones. **Because** churn behavior shifts and random splits would leak future patterns. Define the label window carefully (churn within 30 days of the snapshot) and compute features only from data before the snapshot.

**Model.** Logistic regression baseline, then gradient boosting on usage, tenure, support tickets, and billing features. **Because** tabular, mixed types, non-linear interactions likely.

**Risk.** Predicting churn is not the same as predicting who will respond to an offer. If we can run an experiment, an *uplift* model targets customers whose behavior the offer actually changes, rather than those who would stay anyway.

Case B. Fraud detection, real time, 0.1% positive rate

**Framing.** False negatives cost money; false positives cost customer friction and manual review capacity. Latency is tens of milliseconds.

**Metric.** PR-AUC for model comparison; recall at a fixed review capacity or fixed precision for the deployment target. Cost-weighted thresholding.

**Model.** Gradient-boosted trees with engineered velocity and aggregate features (transactions in the last hour, new device, distance from last location). **Because** they handle imbalanced tabular data well, infer fast, and are easier to monitor than a deep net. Add a rules layer for known hard patterns.

**Pitfalls.** Delayed labels (chargebacks arrive weeks later), so recent data looks cleaner than it is; adversaries adapt, so drift is expected and retraining must be frequent; features must be computable at decision time from the serving store.

**Switch if** relationships among entities matter (rings of accounts): consider graph features.

Case C. Image classifier with 2,000 labeled images

**Choice.** Fine-tune a pretrained convolutional or vision-transformer model rather than training from scratch. **Because** pretrained features transfer; 2,000 images is far too few to learn good representations alone, and scratch training would overfit.

**Plan.** Freeze most layers first and train the head, then unfreeze gradually at a lower learning rate. Use label-preserving augmentation, cross-validation for an honest estimate, and early stopping.

**Checks.** Confusion matrix and per-class metrics; inspect misclassified images for label errors or spurious cues (background, watermark). **Because** small datasets are vulnerable to shortcut features that vanish in production.

Case D. Forecasting weekly demand for 5,000 products

**Baseline.** Seasonal naive (same week last year) and a moving average. **Because** they are hard to beat on stable series and tell me how much structure a model must find.

**Model.** A single global gradient-boosting model across products with lag features, calendar, price, and promotion variables, rather than 5,000 separate models. **Because** it shares strength across series, helps sparse products, and is one pipeline to maintain. Use classical per-series models (ETS, ARIMA) where series are few and long.

**Evaluation.** Rolling-origin backtests; metrics such as MAE or weighted MAPE chosen by business cost (stockouts versus overstock may be asymmetric, so consider quantile forecasts). Never use random k-fold.

Case E. Recommender for a new marketplace (cold start)

**Constraint.** Few interactions per user and item.

**Approach.** Begin with popularity and content-based recommendations using item attributes; add collaborative filtering (matrix factorization or two-tower embeddings) as interactions accumulate. Blend with exploration (a bandit) for new items. **Because** collaborative methods need interaction data, and the system’s own recommendations shape what data it later sees, so some deliberate exploration prevents a popularity feedback loop.

**Metrics.** Offline recall@k and NDCG, but validate with online metrics (click-through, conversion, retention) since offline logs are biased by the previous system’s choices.

Case F. Medical screening model, 5% prevalence

**Framing.** Missing a case is far worse than a false alarm that leads to a confirmatory test, so I would prioritize recall with an acceptable precision, and set the threshold from a stated cost ratio agreed with clinicians.

**Rigor.** Patient-level splits (never the same patient in train and test), external validation on a different site, calibration, subgroup performance, and an explanation clinicians can sanity-check. **Because** models often fail to transfer across hospitals due to differing equipment and populations.

## 10. Deployment and monitoring

Train-serve skew

Features computed one way offline and another way online. Prevent with shared feature code or a feature store, and log served features for later comparison.

Drift

**Data drift**: input distribution changes. **Concept drift**: the relationship between inputs and target changes. Monitor input distributions, prediction distributions, and, when labels arrive, actual performance. **Because** accuracy cannot be measured without labels, distribution monitoring is the early-warning system.

Rollout

Shadow mode (run without acting), then a canary or A/B test with guardrail metrics, with a quick rollback path. **Because** offline gains often shrink or reverse online.

Retraining cadence

Driven by how fast the world changes and how costly stale predictions are: scheduled, triggered by drift, or continual. Always compare a candidate to the current model on the same recent data before promoting.

Feedback loops

If predictions influence the data you collect (who gets a loan, which item is shown), training data becomes biased. Use randomization or exploration, and be careful what you treat as ground truth.

## 11. Rapid-fire questions with crisp answers

Say the answer first in one sentence, then offer the reason. Open each to check yourself.

Why does L1 produce sparse solutions?

The L1 constraint region is a diamond with corners on the axes; the loss contours are most likely to touch it at a corner where some weights are exactly zero. Equivalently, the L1 penalty’s gradient has constant magnitude, so it keeps pushing small weights to zero, while L2’s gradient shrinks as the weight shrinks.

What is the difference between bagging and boosting?

Bagging trains models independently on bootstrap samples and averages them to reduce variance. Boosting trains models sequentially, each focusing on the previous errors, to reduce bias.

Why can ROC-AUC be misleading with imbalanced data?

The false positive rate divides by the large number of negatives, so many false positives still give a small rate. A model can look excellent in ROC space while its precision is poor. PR curves reflect that directly.

What does cross-validation estimate, and what does it not?

It estimates how a training *procedure* generalizes to unseen data from the same distribution, with a variance estimate. It does not protect against leakage, distribution shift, or selection bias from repeatedly tuning on it.

Why standardize features?

For distance-based and gradient-based methods, features on large scales dominate distances and make the loss surface elongated, slowing convergence. Regularization penalties also treat all weights equally, so scale changes who gets penalized. Trees are invariant to monotonic scaling.

What does the kernel trick do?

It computes inner products in a high-dimensional feature space via a kernel function without mapping the data there explicitly, letting linear methods like SVM learn non-linear boundaries.

Explain the curse of dimensionality.

As dimensions grow, data becomes sparse and distances between points concentrate, so nearest neighbors become less meaningful and the sample size needed to cover the space grows exponentially. Remedies: dimensionality reduction, feature selection, regularization, stronger inductive bias.

How does dropout work, and what happens at inference?

During training each unit is dropped with probability p, preventing co-adaptation and acting like averaging an ensemble of sub-networks. At inference all units are used, with activations scaled (or scaled during training in the “inverted” form) to keep expected values consistent.

Why use softmax with cross-entropy?

Softmax converts logits to a probability distribution; cross-entropy is the negative log-likelihood of the correct class. Combined, the gradient with respect to logits is (probability minus one-hot label), which is simple and numerically stable when implemented as a fused log-softmax.

What is the difference between parameters and hyperparameters?

Parameters are learned from data (weights); hyperparameters are set beforehand or tuned on validation data (learning rate, depth, regularization strength).

How do you handle high-cardinality categorical features?

Target encoding fitted out-of-fold to avoid leakage, frequency encoding, hashing, or learned embeddings. One-hot becomes huge and sparse, and trees can overfit to rare categories.

When would accuracy on a balanced dataset still be a poor choice?

When error costs are asymmetric, for example a missed diagnosis versus a false alarm, or when probabilities matter. Then use cost-weighted metrics, recall at a precision target, or log loss.

What is data drift versus concept drift, and how do you detect each?

Data drift is a change in input distribution P(X), detectable without labels via statistical tests or population stability metrics. Concept drift is a change in P(Y|X), usually detected only through degrading performance on newly labeled data or proxy signals.

Why can a more complex model be worse even with enough data?

If labels are noisy or features are weak, flexible models fit noise, and higher complexity also adds optimization difficulty, latency, and monitoring burden. Extra capacity helps only when there is learnable signal beyond what a simpler model captures.

What is the difference between correlation in features and multicollinearity’s effect?

Correlated features make linear model coefficients unstable and hard to interpret (large variance), though predictions can remain fine. Ridge or elastic net stabilize them; tree models are less affected for prediction but split importance gets divided among correlated features.

How would you explain precision and recall to a non-technical stakeholder?

Precision: when the system raises an alert, how often is it right. Recall: of all the real problems, how many does it catch. Raising the alert bar improves the first and hurts the second, so we choose the balance by comparing the cost of a false alarm with the cost of a miss.

## 12. Common traps interviewers probe

- **Tuning on the test set,** or reporting the best of many experiments without a fresh holdout.
- **Fitting preprocessing on all data** before splitting.
- **Random splits on time series** or grouped data.
- **Using accuracy on imbalanced data** and celebrating the number.
- **Rebalancing the test set,** then claiming production performance.
- **Treating feature importance as causation.** Importance describes what the model uses, not what causes the outcome.
- **Ignoring baselines.** Without one, no result can be judged.
- **Ignoring costs.** Training cost, inference latency, and maintenance burden are part of “best model.”
- **Over-claiming from small differences** without confidence intervals.
- **Ignoring how the model will be used,** including who acts on the prediction and what happens on a wrong one.

**Turn a trap into a strength.** When you mention a trap you have avoided (“I would fit the scaler inside the cross-validation pipeline to avoid leakage”), you demonstrate experience, because these are exactly the mistakes people make on real projects.

## 13. Cheat sheet

### Formulas to recall

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)         (sensitivity, true positive rate)
Specificity = TN / (TN + FP)
F1        = 2PR / (P + R)          F-beta = (1+b^2) PR / (b^2 P + R)

Log loss  = -(1/N) sum[ y log p + (1-y) log(1-p) ]
MSE       = (1/N) sum (y - y_hat)^2        RMSE = sqrt(MSE)
Bias-variance (squared loss): E[err] = bias^2 + variance + noise

Logistic: p = 1 / (1 + e^{-z})            dL/dz = p - y   (with log loss)
Softmax:  p_i = e^{z_i} / sum_j e^{z_j}
Ridge:    argmin ||y - Xw||^2 + lambda ||w||_2^2
Lasso:    argmin ||y - Xw||^2 + lambda ||w||_1
Cost-optimal threshold (calibrated p) = C_FP / (C_FP + C_FN)

SGD:      w = w - lr * g
Momentum: v = b v + g ;  w = w - lr * v
Adam:     m = b1 m + (1-b1) g ; v = b2 v + (1-b2) g^2 ;
          w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

### Which metric, in one line

- Balanced and symmetric costs: accuracy or F1.
- Rare positives: PR-AUC and recall at fixed precision.
- Need probabilities: log loss, Brier, calibration curve.
- Outliers present: MAE or Huber; large errors extra bad: RMSE.
- Ranking: NDCG, MAP, recall@k.

### Which model, in one line

- Need a baseline or interpretability: logistic or linear regression.
- Tabular accuracy: gradient boosting.
- Robust, low tuning: random forest.
- Unstructured data (image, text, audio): pretrained neural network, fine-tuned.
- Few labels: pretrained features, regularization, simpler models, careful CV.

## 14. Practice plan

| Step | What to do | Why it helps |
| --- | --- | --- |
| 1 | Take one tabular dataset and build the full pipeline: baseline, leakage-safe split, tuned boosted model, calibrated probabilities, threshold chosen from a cost matrix. | Exercises every decision in sections 3 to 6 once, end to end. |
| 2 | Deliberately break it: introduce a leaky feature and a random split on time data, and see the inflated scores. | You will recognize leakage instantly in an interview and can describe it from experience. |
| 3 | Pick ten questions from sections 9 and 11, answer each aloud in under two minutes using the section 1 skeleton, and record yourself. | Decision reasoning is a speaking skill; practice it that way. |
| 4 | For each model in section 4, write one sentence on when you would *not* use it. | Knowing a tool’s limits is what separates judgment from memorization. |
| 5 | Take a past project and write a one-page “what I would do differently and why.” | Gives you honest, specific stories for behavioral and deep-dive rounds. |

### Questions to ask yourself before the interview

- Can I explain bias-variance, regularization, and cross-entropy without notes and connect each to a decision?
- For any metric I name, can I say when it would mislead?
- For any model I recommend, can I say what would make me change my mind?
- Do I have two stories where I caught leakage, drift, or a bad metric choice?

### Questions you can ask the interviewer

- How do you decide whether a model is good enough to ship, and who owns that call?
- How do you monitor models after launch, and how quickly do you find out when they degrade?
- What did a recent model failure teach the team?

Prepared November 2026. Specific techniques and library names evolve; the underlying reasoning patterns are the durable part.