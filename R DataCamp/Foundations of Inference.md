---
typora-copy-images-to: ./figures
---



Working with the NHANES data
===

Throughout this chapter, you will use the `NHANES` dataset from the `NHANES` R package. These data are collected by the CDC and can be thought of as a random sample of US residents.

Before moving on to investigate particular variables, you'll have an opportunity to briefly explore the data in this exercise.

```R
# Load packages
library(dplyr)
library(ggplot2)
library(NHANES)


# What are the variables in the NHANES dataset?
names(NHANES)

# Create bar plot for Home Ownership by Gender
ggplot(NHANES, aes(x = Gender, fill = HomeOwn)) + 
  geom_bar(position = "fill") +
  ylab("Relative frequencies")

# Density for SleepHrsNight colored by SleepTrouble, faceted by HealthGen
ggplot(NHANES, aes(x = SleepHrsNight, col = SleepTrouble)) + 
  geom_density(adjust = 2) + 
  facet_wrap(~ HealthGen)
```

Randomly allocating samples
===

As seen in the video, you will now use R to randomly permute the observations and calculate a difference in proportions that could arise from a null distribution.

Using the `NHANES` dataset, let's investigate the relationship between gender and home ownership. 

- Subset the `NHANES` dataset to consider only individuals whose home ownership status is either `"Own"` or `"Rent"`. Save the result to `homes`.
- Perform a single permutation to evaluate whether home ownership status (i.e. `HomeOwn`) differs between the `"female"` and `"male"`groups:
- In your `mutate()` call, use `sample()` to shuffle home ownership status. Call this new variable `HomeOwn_perm`, a permuted version of `HomeOwn`.
- Group by `Gender`.
- In both the original data and in the permuted data, compute the proportion of individuals who own a home. Note that this will calculate proportions for both genders since you've grouped by the `Gender` variable in the line before it.
- Using the `diff()` function, calculate the difference in proportion of home ownership for both `prop_own_perm`, the permuted data, and `prop_own`, the original data.

```R
# Subset the data: homes
homes <- NHANES %>%
  select(Gender, HomeOwn) %>%
  filter(HomeOwn %in% c("Own", "Rent"))

# Perform one permutation 
homes %>%
  mutate(HomeOwn_perm = sample(HomeOwn)) %>%
  group_by(Gender) %>%
  summarize(prop_own_perm = mean(HomeOwn_perm == "Own"), 
            prop_own = mean(HomeOwn == "Own")) %>%
  summarize(diff_perm = diff(prop_own_perm),
            diff_orig = diff(prop_own))
```

Randomization dotplot (n = 10)
===

As you saw in the video, natural variability can be modeled from shuffling observations around to remove any relationships that might exist in the population. This is done with the `rep_sample_n()` function from the `oilabs` package. Within it, you must specify arguments for the data (`tbl`), the sample `size`, the number of samples to take (`reps`), and whether sampling should be done with or without replacement (`replace`). The output includes a new column, `replicate`, which indicates the sample number. For example,

```
homes %>% 
  rep_sample_n(size = 5, reps = 3)

```

will return three samples of 5 observations from the `homes` dataset you created in the last exercise. The first 5 rows will have a value of 1 in the `replicate` column, the next 5 rows will have a value of 2, and so on. Note that the default value for the `replace` argument is `FALSE`.

The `rep_sample_n()` function is useful here because it adds the `replicate` column. This ensures that you can keep many different random samples in one table without getting them confused. For example, grouping by the values in the `replicate` column and using a `summarise()` call lets you effectively do a calculation on every one of the shuffled datasets that you've made. You can see why this makes our lives easier when we're making more than one or two samples!

In this exercise, you will permute the home ownership variable 10 times. By doing so, you will ensure that there is no relationship between home ownership and gender, so any difference in home ownership proportion for female versus male will be due only to natural variability.

```R
# Perform 10 permutations
homeown_perm <- homes %>%
  rep_sample_n(size = nrow(homes), reps = 10) %>%
  mutate(HomeOwn_perm = sample(HomeOwn)) %>%
  group_by(replicate, Gender) %>%
  summarize(prop_own_perm = mean(HomeOwn_perm == "Own"), 
            prop_own = mean(HomeOwn == "Own")) %>%
  summarize(diff_perm = diff(prop_own_perm),
            diff_orig = diff(prop_own)) # male - female

# Print differences to console
print(homeown_perm)

# Dotplot of 10 permuted differences in proportions
ggplot(homeown_perm, aes(x = diff_perm)) + 
  geom_dotplot(binwidth = .001)
```

![1C7F9CD1-724A-4FA4-960C-F22B8AD0A55A](figures/1C7F9CD1-724A-4FA4-960C-F22B8AD0A55A.png)

From the dotplot, it's still hard to get an idea of what the null distribution looks like. You'll repeat the process with 100 permutations.

![A308A221-53E9-46EF-A147-FE99149A0648](figures/A308A221-53E9-46EF-A147-FE99149A0648.png)

Nice! It's much easier to get a feel for the distribution now. In the next exercise, you'll see how 1000 permutations are distributed.

![3666754F-9AA2-44F5-8E0A-298F5CD36C1C](figures/3666754F-9AA2-44F5-8E0A-298F5CD36C1C.png)

You can now see that the distribution is approximately normally distributed around -0.01, but what can we conclude from it? 

Do the data come from the population?
===

Recall that the observed difference (i.e. the difference in proportions in the `homes` dataset, shown as the red vertical line) was around -0.0078, which seems to fall below the bulk of the density of shuffled differences. It is important to know, however, whether any of the randomly permuted differences were as extreme as the observed difference.

```R
# Plot permuted differences
ggplot(homeown_perm, aes(x = diff_perm)) + 
  geom_density() +
  geom_vline(aes(xintercept = diff_orig), col = "red")

# Compare permuted differences to observed difference
homeown_perm %>%
  summarize(sum(diff_orig >= diff_perm))
```

![FB25F2A2-4451-4222-B79F-99F189181141](figures/FB25F2A2-4451-4222-B79F-99F189181141.png)

212 permuted differences are more extreme than the observed difference. This only represents 21.2% of the null statistics, so you can conclude that the observed difference **is** consistent with the permuted distribution.

### What can you conclude from the analysis?

**We have learned that our data is consistent with the hypothesis of no difference in home ownership across gender and Right! We've failed to reject the null hypothesis**

---

Summarizing gender discrimination
===

As the first step of any analysis, you should look at and summarize the data. Categorical variables are often summarized using proportions, and it is always important to understand the denominator of the proportion.

Do you want the proportion of women who were promoted or the proportion of promoted individuals who were women? Here, you want the first of these, so in your R code it's necessary to `group_by()` the `sex` variable.

```R
# Create a contingency table summarizing the data
table(disc)
              sex
promote        female male
  not_promoted     10    3
  promoted         14   21

# Find proportion of each sex who were promoted
disc %>%
  group_by(sex) %>%
  summarize(promoted_prop = mean(promote == "promoted"))
# A tibble: 2 × 2
     sex promoted_prop
  <fctr>         <dbl>
1 female     0.5833333
2   male     0.8750000
```

so the difference in proportions promoted is almost 0.3.

Step-by-step through the permutation
===

To help you understand the code used to create the randomization distribution, this exercise will walk you through each step, one at a time.

Remember that when using the pipe notation (`%>%`) with so-called *tidy*functions like `mutate()` and `summarize()` from the `dplyr`package, each input is a data frame and each output is also a data frame. To keep the output manageable, we will use only 5 replicates here.

```R
# Sample the entire data frame 5 times
disc %>%
  rep_sample_n(size = nrow(disc), reps = 5) 

# Shuffle the promote variable within replicate
disc %>%
  rep_sample_n(size = nrow(disc), reps = 5) %>%
  mutate(prom_perm = sample(promote)) 

# Find the proportion of promoted in each replicate and sex
disc %>%
  rep_sample_n(size = nrow(disc), reps = 5) %>%
  mutate(prom_perm = sample(promote)) %>%
  group_by(replicate, sex) %>%
  summarize(prop_prom_perm = mean(prom_perm == "promoted"),
            prop_prom = mean(promote == "promoted")) 

# Difference in proportion of promoted across sex grouped by gender
disc %>%
  rep_sample_n(size = nrow(disc), reps = 5) %>%
  mutate(prom_perm = sample(promote)) %>%
  group_by(replicate, sex) %>%
  summarize(prop_prom_perm = mean(prom_perm == "promoted"),
            prop_prom = mean(promote == "promoted"))  %>%
  summarize(diff_perm = diff(prop_prom_perm),
            diff_orig = diff(prop_prom))  # male - female
```

Let's quickly recap what you just did. Using `rep_sample_n()`, you took 5 repeated samples (i.e. Replications) of the `disc` data, then shuffled these using `sample()` to break any links between gender and getting promoted. Then for each replication, you calculated the proportions of promoted males and females in the dataset along with the difference in proportions.

Randomizing gender discrimination
===

Recall that we are considering a situation where the number of men and women are fixed (representing the resumes) and the number of people promoted is fixed (the managers were able to promote only 35 individuals).

In this exercise, you'll create a randomization distribution of the null statistic with 1000 replicates as opposed to just 5 in the previous exercise. As a reminder, the statistic of interest is the difference in proportions promoted between genders (i.e. proportion for males minus proportion for females).

```R
# Create a data frame of differences in promotion rates
disc_perm <- disc %>%
  rep_sample_n(size = nrow(disc), reps = 1000) %>%
  mutate(prom_perm = sample(promote)) %>%
  group_by(replicate, sex) %>%
  summarize(prop_prom_perm = mean(prom_perm == "promoted"),
            prop_prom = mean(promote == "promoted")) %>%
  summarize(diff_perm = diff(prop_prom_perm),
            diff_orig = diff(prop_prom))  # male - female


# Histogram of permuted differences
ggplot(disc_perm, aes(x = diff_perm)) + 
  geom_histogram(binwidth = .01) +
  geom_vline(aes(xintercept = diff_orig), col = "red")
```

![E6155E10-3A27-4120-835A-12FC283EAFA2](figures/E6155E10-3A27-4120-835A-12FC283EAFA2.png)

Due to above picture, we can say: In the population there is evidence that women are promoted at a different rate, but we cannot tell whether the difference is due to discrimination or something else.

Critical region
===

It seems as though the statistic—a difference in promotion rates of 0.2917—is on the extreme end of the permutation distribution. That is, there are very few permuted differences which are as extreme as the observed difference.

To quantify the extreme permuted (null) differences, we use the `quantile()` function.

```R
# Find the 0.90, 0.95, and 0.99 quantiles of diff_perm
disc_perm %>% 
  summarize(q.90 = quantile(diff_perm, p = .90),
            q.95 = quantile(diff_perm, p = .95),
            q.99 = quantile(diff_perm, p = .99))
# A tibble: 1 × 3
       q.90      q.95      q.99
      <dbl>     <dbl>     <dbl>
1 0.2083333 0.2083333 0.2916667
```

Two-sided critical region
===

For the discrimination data, the question at hand is whether or not women were promoted less often than men. However, there are often scenarios where the research question centers around a difference without directionality.

For example, you might be interested in whether the rate of promotion for men and women is *different*. In that case, a difference in proportions of -0.29 is just as "extreme" as a difference of positive 0.29.

If you had seen that women were promoted more often, what would the other side of the distribution of permuted differences look like? That is, what are the smallest (negative) values of the distribution of permuted differences?

```R
# Find the 0.01, 0.05, and 0.10 quantiles of diff_perm
disc_perm %>% 
  summarize(q.01 = quantile(diff_perm, p=.01),
            q.05 = quantile(diff_perm, p=.05),
            q.10 = quantile(diff_perm, p=.10))
        q.01       q.05   q.10
       <dbl>      <dbl>  <dbl>
1 -0.2916667 -0.2083333 -0.125
```

How does sample size affect results?
===

Notice that the observed difference of 0.2917 is in the extreme right tail of the permuted differences. If the **sample was ten times larger** but the sample statistic was exactly the same (i.e. 0.2917), how would the distribution of permuted differences change?

**Be much farther to the right of the permuted differences (completely off of the distribution).**

Sample size in randomization distribution
===

We've created two new datasets for you with essentially the same difference in proportions as the original discrimination data. However, one of the datasets (`disc_small`) is one third the size of the original dataset and the other (`disc_big`) is 10 times larger than the original dataset.

Additionally, the same permutation code used previously has been run on the small and big datasets to create small and big distributions of permuted differences in promotion rates (`disc_small_perm` and `disc_big_perm`, respectively).

In this exercise, you'll use these two new distributions to get a sense for how the differences vary given widely different sample sizes. In particular, notice the range of variability on the x-axis of each plot.

```R
# Tabulate the small and big data frames
disc_small %>% 
  select(sex, promote) %>%
  table()

##       promote
sex      not_promoted promoted
  female            3        5
  male              1        7


disc_big %>% 
  select(sex, promote) %>%
  table()

##       promote
sex      not_promoted promoted
  female          100      140
  male             30      210

# Plot the distributions of permuted differences
ggplot(disc_small_perm, aes(x = diff_perm)) + 
  geom_histogram(binwidth = .01) +
  geom_vline(aes(xintercept = diff_orig), col = "red")

ggplot(disc_big_perm, aes(x = diff_perm)) + 
  geom_histogram(binwidth = .01) +
  geom_vline(aes(xintercept = diff_orig), col = "red")
```

![DA01CA7D-6270-4BEA-AD68-D0DEBD376DE3](figures/DA01CA7D-6270-4BEA-AD68-D0DEBD376DE3.png)

![946F119D-1287-448B-987E-80CEC89BA947](figures/946F119D-1287-448B-987E-80CEC89BA947.png)

The observed difference is consistent with differences you would see by chance if the sample size was small. The observed difference would virtually never be observed by chance if the sample size was big.

Sample size for critical region
===

Using the randomization distributions with the small and big datasets, calculate different cutoffs for significance. Remember, you are most interested in a large positive difference in promotion rates, so you are calculating the upper quantiles of 0.90, 0.95, and 0.99.

```R
# Recall the quantiles associated with the original dataset
disc_perm %>% 
  summarize(q.90 = quantile(diff_perm, p = 0.90),
            q.95 = quantile(diff_perm, p = 0.95),
            q.99 = quantile(diff_perm, p = 0.99))
# A tibble: 1 × 3
       q.90      q.95      q.99
      <dbl>     <dbl>     <dbl>
1 0.2083333 0.2083333 0.2916667

# Calculate the quantiles associated with the small dataset
disc_small_perm %>% 
  summarize(q.90 = quantile(diff_perm, p = 0.90),
            q.95 = quantile(diff_perm, p = 0.95),
            q.99 = quantile(diff_perm, p = 0.99))
# A tibble: 1 × 3
   q.90  q.95  q.99
  <dbl> <dbl> <dbl>
1  0.25  0.25   0.5

# Calculate the quantiles associated with the big dataset
disc_big_perm %>% 
  summarize(q.90 = quantile(diff_perm, p = 0.90),
            q.95 = quantile(diff_perm, p = 0.95),
            q.99 = quantile(diff_perm, p = 0.99))
# A tibble: 1 × 3
   q.90    q.95  q.99
  <dbl>   <dbl> <dbl>
1  0.05 0.05875   0.1
```

Notice how the differences in proportions must be much larger to be significant if the sample size is small. With a big sample size, a small difference in proportions can be significant.

Calculating the p-values
===

A p-value measures the degree of disagreement between the data and the null hypothesis. Here, you will calculate the p-value for the original discrimination dataset as well as the small and big versions, `disc_small` and `disc_big`.

Recall that you're only interested in the one-sided hypothesis test here. That is, you're trying to answer the question, "Are men more likely to be promoted than women?"

```R
# Calculate the p-value for the original dataset
disc_perm %>%
  summarize(mean(diff_orig <= diff_perm))
# A tibble: 1 × 1
  `mean(diff_orig <= diff_perm)`
                           <dbl>
1                           0.03

# Calculate the p-value for the small dataset
disc_small_perm %>%
  summarize(mean(diff_orig <= diff_perm))
# A tibble: 1 × 1
  `mean(diff_orig <= diff_perm)`
                           <dbl>
1                          0.277

# Calculate the p-value for the big dataset
disc_big_perm %>%
  summarize(mean(diff_orig <= diff_perm))
# A tibble: 1 × 1
  `mean(diff_orig <= diff_perm)`
                           <dbl>
1                              0
```

Practice calculating p-values
===

In the original dataset, 87.5% of the men were promoted and 58.3% of the women were promoted.

Consider a situation where there are 24 men, 24 women, and 35 people are still promoted. But in this new scenario, 75% of the men are promoted and 70.8% of the women are promoted. Does the difference in promotion rates still appear to be statistically significant? That is, could this difference in promotion rates have come from random chance?

You'll analyze these new data, contained in `disc_new`, using the same permutation algorithm from before.

```R
# Recall the original data
disc %>% 
  select(sex, promote) %>%
  table()

# Tabulate the new data
disc_new %>% 
  select(sex, promote) %>%
  table()

# Plot the distribution of the original permuted differences
ggplot(disc_perm, aes(x = diff_perm)) + 
  geom_histogram() +
  geom_vline(aes(xintercept = diff_orig), col = "red")

# Plot the distribution of the new permuted differences
ggplot(disc_new_perm, aes(x = diff_perm)) + 
  geom_histogram() +
  geom_vline(aes(xintercept = diff_orig), col = "red")



# Find the p-value from the original data
disc_perm %>%
  summarize(mean(diff_orig <= diff_perm))
# A tibble: 1 × 1
  `mean(diff_orig <= diff_perm)`
                           <dbl>
1                           0.03

# Find the p-value from the new data
disc_new_perm %>%
  summarize(mean(diff_orig <= diff_perm))
# A tibble: 1 × 1
  `mean(diff_orig <= diff_perm)`
                           <dbl>
1                          0.497
```

![D7E00AF2-C0F8-4903-837A-FFCBEDEBFB4B](figures/D7E00AF2-C0F8-4903-837A-FFCBEDEBFB4B.png)

![801BDDE7-3CB8-4184-A6AB-F9F747C40F8A](figures/801BDDE7-3CB8-4184-A6AB-F9F747C40F8A.png)

Calculating two-sided p-values
===

What if the original research hypothesis had focused on *any* difference in promotion rates between men and women instead of focusing on whether men are more likely to be promoted than women? In this case, a difference like the one observed would occur twice as often (by chance) because sometimes the difference would be positive and sometimes it would be negative.

When there is no directionality to the alternative hypothesis, the hypothesis and p-value are considered to be *two-sided*. In a two-sided setting, the p-value is double the one-sided p-value.

```R
# Calculate the two-sided p-value
disc_perm %>%
  summarize(2 * mean(diff_orig <= diff_perm))
# A tibble: 1 × 1
  `2 * mean(diff_orig <= diff_perm)`
                               <dbl>
1                               0.06
```

---

Summarizing opportunity cost (1)
===

As you saw in the video, we're interested in whether the treatment and control groups were equally likely to buy a DVD after reading the experimental statements.

In this exercise, you'll use the data from the study to find the sample statistics (here: proportions) that are needed for the analysis.

```R
# Tabulate the data
opportunity %>%
  select(decision, group) %>%
  table()
##         group
decision   control treatment
  buyDVD        56        41
  nobuyDVD      19        34

# Find the proportion who bought the DVD in each group
opportunity %>%
  group_by(group) %>%
  summarize(buy_prop = mean(decision == "buyDVD"))
# A tibble: 2 × 2
      group  buy_prop
     <fctr>     <dbl>
1   control 0.7466667
2 treatment 0.5466667
```

Okay, so about 75% of the control group bought the DVD and about 55% of the treatment group (i.e. The group that was reminded that the money could be saved) bought the DVD.

Plotting opportunity cost
===

Again, interest is in whether the treatment and control groups were equally likely to buy a DVD after reading the experimental statements. Here, you'll create a barplot to visualize the difference in proportions between the treatment and control groups.

```R
# Create a barplot
ggplot(opportunity, aes(x = group, fill = decision)) + 
  geom_bar(position = "fill")
```

![36706168-0AAD-4260-83AA-79709EA66847](figures/36706168-0AAD-4260-83AA-79709EA66847.png)

The barplot better displays the results from the study. The treatment seems like it might have had an effect!

Randomizing opportunity cost
===

As in Chapter 2, you will permute the data to generate a distribution of differences as if the null hypothesis were true.

In the study, the number of individuals in each of the control and treatment groups is fixed. Additionally, when you assume that the null hypothesis is true—that is, the experiment had no effect on the outcome of buying a DVD—it is reasonable to infer that the number of individuals who would buy a DVD is also fixed. That is, 97 people were going to buy a DVD regardless of which treatment group they were in.

Using the new data and the methods from the previous chapter, create a randomization distribution of the difference in proportions calculated on permuted data.

```R
# Data frame of differences in purchase rates after permuting
opp_perm <- opportunity %>%
  rep_sample_n(size = nrow(opportunity), reps = 1000) %>%
  mutate(dec_perm = sample(decision)) %>%
  group_by(replicate, group) %>%
  summarize(prop_buy_perm = mean(dec_perm == "buyDVD"),
            prop_buy = mean(decision == "buyDVD")) %>%
  summarize(diff_perm = diff(prop_buy_perm),
            diff_orig = diff(prop_buy))  # treatment - control


# Histogram of permuted differences
ggplot(opp_perm, aes(x = diff_perm)) + 
  geom_histogram(binwidth = .005) +
  geom_vline(aes(xintercept = diff_orig), col = "red")
```

![056775D1-021B-480B-AF84-9DF50643FC8B](figures/056775D1-021B-480B-AF84-9DF50643FC8B.png)

Summarizing opportunity cost (2)
===

Now that you've created the randomization distribution, you'll use it to assess whether the observed difference in proportions is consistent with the null difference. You will measure this consistency (or lack thereof) with a p-value, or the *proportion of permuted differences less than or equal to the observed difference*.

```R
# Calculate the p-value
opp_perm %>%
  summarize(mean(diff_perm <= diff_orig))
# A tibble: 1 × 1
  `mean(diff_perm <= diff_orig)`
                           <dbl>
1                          0.007
```

Opportunity cost conclusion
===

In the last exercise, you computed the p-value, or the proportion of permuted differences less than or equal to the observed difference:

```R
opp_perm %>%
  summarize(mean(diff_perm <= diff_orig))
```

Based on this result of 0.007, what can you conclude from the study about the effect of reminding students to save money?

We can confidently say the different messaging *caused*the students to change their buying habits, since they were randomly assigned to treatment and control groups.

p-value for two-sided hypotheses: opportunity costs
===

The p-value measures the likelihood of data as or more extreme than the observed data, given the null hypothesis is true. Therefore, the appropriate p-value for a two-sided alternative hypothesis is a two-sided p-value.

To find a two-sided p-value, you simply double the one sided p-value. That is, you want to find two times the proportion of permuted differences that are less than or equal to the observed difference.

```R
# Calculate the two-sided p-value
opp_perm %>%
  summarize(2 * mean(diff_perm <= diff_orig))
```

---

What is the parameter?
===

In November 2016, the voters elected a new president of the United States. Prior to the election, thousands of polls were taken to gauge the popularity of each of the candidates. Leaving aside the idea that popular opinion changes over time, a poll can be thought of as a sample of individuals measured so as to estimate the proportion of all voters who will vote for each candiate (i.e. the population parameter).

Consider an election in your home town that will take place in a week's time. You poll a randomly selected subset of the voters in your town and ask them if they plan to vote for Candidate X or Candidate Y. In this chapter, we will focus on *sampling variability*—the variability in sample proportions due to polling different randomly selected individuals from the population.

Before investigating the sampling variability, what is the population parameter of interest?

**The proportion of all voters in your town who will vote for Candidate X on election day.**

Hypothesis test or confidence interval?
===

A university is trying to determine whether parking is a problem on its campus. The student newspaper contacts a random sample of 200 students and asks whether or not they are frustrated with the parking situation. They want to estimate the proportion of students at the college who are frustrated with the parking situation.

In this setting, which is more appropriate, a hypothesis test or a confidence interval?

**Confidence interval because the goal is to estimate a population parameter.**

Resampling from a sample
===

To investigate how much estimates of a population proportion change from sample to sample, you will set up two sampling experiments.

In the first experiment, you will simulate repeated samples from a population. In the second, you will choose a single sample from the first experiment and repeatedly resample from that sample—a method called *bootstrapping*. More specifically:

**Experiment 1**: Assume the true proportion of people who will vote for Candidate X is 0.6. Repeatedly sample 30 people from the population and measure the variability of $\hat p$ (the *sample* proportion).

**Experiment 2**: Take *one* sample of size 30 from the same population. Repeatedly sample 30 people (with replacement!) from the original sample and measure the variability of $\hat p *$ (the *resample* proportion).

It's important to realize that the first experiment relies on knowing the population and is **typically impossible in practice**. The second relies only on the sample of data and is therefore easy to implement for any statistic. Fortunately, as you will see, the variability in p̂, or the proportion of "successes" in a sample, is approximately the same whether we sample from the population or resample from a sample.

```R
# Select one poll from which to resample: one_poll
one_poll <- all_polls %>%
  filter(poll == 1) %>%
  select(vote)
  
# Generate 1000 resamples of one_poll: one_poll_boot_30
one_poll_boot_30 <- one_poll %>%
  rep_sample_n(size = 30, replace = TRUE, reps = 1000)

# Compute p-hat for each poll: ex1_props
ex1_props <- all_polls %>% 
  group_by(poll) %>% 
  summarize(prop_yes = mean(vote))
  
# Compute p-hat* for each resampled poll: ex2_props
ex2_props <- one_poll_boot_30 %>%
  summarize(prop_yes = mean(vote))

# Compare variability of p-hat and p-hat*
ex1_props %>% summarize(sd(prop_yes))
# A tibble: 1 × 1
  `sd(prop_yes)`
           <dbl>
1     0.08683128
ex2_props %>% summarize(sd(prop_yes))
# A tibble: 1 × 1
  `sd(prop_yes)`
           <dbl>
1      0.0883612
```

The variability in the proportion of “successes” in a sample is approximately the same whether we sample from the population or resample from a sample.

Resampling from a sample (2)
===

In the previous exercise, the resamples (with replacement) were the same size as the original dataset. You originally polled 30 people, then you repeatedly resampled 30 votes from the original dataset (with replacement).

What if the original dataset was 30 observations, but you chose to resample only 3 individuals with replacement? Alternatively, what if you chose to resample 300 individuals with replacement? Let's call these **Experiment 3** and **Experiment 4**, respectively.

Would the variability in these *resampled* p̂ ∗p^∗ values still be a good proxy for the variability of the *sampled* p̂ p^ values taken from repeated samples from the population?

```R
# Resample from one_poll with n = 3: one_poll_boot_3
one_poll_boot_3 <- one_poll %>%
  rep_sample_n(size = 3, replace = TRUE, reps = 1000)

# Resample from one_poll with n = 300: one_poll_boot_300
one_poll_boot_300 <- one_poll %>%
  rep_sample_n(size = 300, replace = TRUE, reps = 1000)
  
# Compute p-hat* for each resampled poll: ex3_props
ex3_props <- one_poll_boot_3 %>% 
  summarize(prop_yes = mean(vote))
  
# Compute p-hat* for each resampled poll: ex4_props
ex4_props <- one_poll_boot_300 %>% 
  summarize(prop_yes = mean(vote))

# Compare variability of p-hat* for n = 3 vs. n = 300
ex3_props %>% summarize(sd(prop_yes))
# A tibble: 1 × 1
  `sd(prop_yes)`
           <dbl>
1      0.2624964
ex4_props %>% summarize(sd(prop_yes))
# A tibble: 1 × 1
  `sd(prop_yes)`
           <dbl>
1     0.02661213
```

Recall the variability of the sampled values taken from repeated samples from the population was about 0.0868. Note that resampling 3 or 300 individuals with replacement from the 30-observation dataset is **not** a good approximation of this value. 

Visualizing the variability of p-hat
===

In order to compare the variability of the sampled p̂ p^ and p̂ ∗p^∗ values in the previous exercises, it is valuable to visualize their distributions. To recall, the exercises walked through four different experiments for investigating the variability of p̂ and p̂∗:

- **Experiment 1**: Sample (n=30n=30) repeatedly from an extremely large population (gold standard, but unrealistic)
- **Experiment 2**: Resample (n=30n=30) repeatedly with replacement from a single sample of size 30
- **Experiment 3**: Resample (n=3n=3) repeatedly with replacement from a single sample of size 30
- **Experiment 4**: Resample (n=300n=300) repeatedly with replacement from a single sample of size 30

```R
# Recall the variability of sample proportions
ex1_props %>% summarize(sd(prop_yes))
ex2_props %>% summarize(sd(prop_yes))
ex3_props %>% summarize(sd(prop_yes))
ex4_props %>% summarize(sd(prop_yes))

# Create smoothed density curves for all four experiments
ggplot() + 
  geom_density(data = ex1_props, aes(x = prop_yes), col = "black", bw = .1) +
  geom_density(data = ex2_props, aes(x = prop_yes), col = "green", bw = .1) +
  geom_density(data = ex3_props, aes(x = prop_yes), col = "red", bw = .1) +
  geom_density(data = ex4_props, aes(x = prop_yes), col = "blue", bw = .1)
```

![9449E1D3-2A35-41C8-A5B7-E89EE92424D4](figures/9449E1D3-2A35-41C8-A5B7-E89EE92424D4.png)

Empirical Rule
===

Many statistics we use in data analysis (including both the sample average and sample proportion) have nice properties that are used to better understand the population parameter(s) of interest.

One such property is that if the variability of the sample proportion (called the *standard error*, or SE) is known, then approximately 95% of p̂ values (from different samples) will be within 2SE of the true population proportion.

To check whether that holds in the situation at hand, let's go back to the polls generated by taking many samples from the same population.

```R
# Compute proportion of votes for Candidate X: props
props <- all_polls %>%
  group_by(poll) %>% 
  summarize(prop_yes = mean(vote))

# Proportion of polls within 2SE
props %>%
  mutate(lower = 0.6 - 2 * sd(prop_yes),
         upper = 0.6 + 2 * sd(prop_yes),
         in_CI = prop_yes > lower & prop_yes < upper) %>%
  summarize(mean(in_CI))
# A tibble: 1 × 1
  `mean(in_CI)`
          <dbl>
1         0.966
```

 In this example, it looks like 96.6% are within 2 standard errors of the true population parameter.

Bootstrap t-confidence interval
===

The previous exercises told you two things:

1. You can measure the variability associated with p̂  by resampling from the original sample.
2. Once you know the variability of p̂ , you can use it as a way to measure how far away the true proportion is.

Note that the *rate* of closeness (here 95%) refers to how often a sample is chosen so that it is close to the population parameter. You won't ever know if a particular dataset is close to the parameter or far from it, but you do know that over your lifetime, 95% of the samples you collect should give you estimates that are within 2SE of the true population parameter.

```R
# Again, set the one sample that was collected
one_poll <- all_polls %>%
  filter(poll == 1) %>%
  select(vote)
  
# Compute p-hat from one_poll: p_hat
p_hat <- mean(one_poll$vote)

# Bootstrap to find the SE of p-hat: one_poll_boot
one_poll_boot <- one_poll %>%
  rep_sample_n(30, replace = TRUE, reps = 1000) %>%
  summarize(prop_yes_boot = mean(vote))

# Create an interval of plausible values
one_poll_boot %>%
  summarize(lower = p_hat - 2 * sd(prop_yes_boot),
            upper = p_hat + 2 * sd(prop_yes_boot))
# A tibble: 1 × 2
      lower     upper
      <dbl>     <dbl>
1 0.5318633 0.8681367
```

Remember that a confidence level describes how likely you are to have gotten a sample that was close enough to the true parameter. Indeed, with the sample at hand, the confidence interval of plausible values *does* contain the true population parameter of 0.6. 

Bootstrap percentile interval
===

The main idea in the previous exercise was that the distance between the original sample p̂ p^ and the resampled (or bootstrapped) p̂ ∗p^∗ values gives a measure for how far the original p̂ p^ is from the true population proportion.

The same variability can be measured through a different mechanism. As before, if p̂ p^ is sufficiently close to the true parameter, then the resampled (bootstrapped) p̂ ∗p^∗ values will vary in such a way that they overlap with the true parameter.

Instead of using ±2SE±2SE as a way to measure the middle 95% of the sampled p̂ p^ values, you can find the middle of the resampled p̂ ∗p^∗ values by removing the upper and lower 2.5%. Note that this second method of constructing bootstrap intervals also gives an intuitive way for making 90% or 99% confidence intervals.

```R
# Find the 2.5% and 97.5% of the p-hat values
one_poll_boot %>% 
  summarize(q025_prop = quantile(prop_yes_boot, p = .025),
            q975_prop = quantile(prop_yes_boot, p = .975))

# Bootstrap t-confidence interval for comparison
one_poll_boot %>%
  summarize(lower = p_hat - 2 * sd(prop_yes_boot),
            upper = p_hat + 2 * sd(prop_yes_boot))
```

Note that the two intervals were created using different methods. Because the methods are different, the intervals are expected to be a bit different as well. In the long run, however, the intervals should provide the same information.

Sample size effects on bootstrap CIs
===

In a previous exercise, you found that if you resampled the data with the wrong size (e.g. 300 or 3 instead of 30), the standard error of the sample proportions was off.

Here, you will use the *incorrect* standard error (based on the incorrect sample size) to create a confidence interval. The idea is that when the standard error is off, the interval is not particularly useful, nor is it correct.

```R
# Recall the bootstrap t-confidence interval
p_hat <- mean(one_poll$vote)
one_poll_boot %>%
  summarize(lower = p_hat - 2 * sd(prop_yes_boot),
            upper = p_hat + 2 * sd(prop_yes_boot))
# A tibble: 1 × 2
     lower    upper
     <dbl>    <dbl>
1 0.536148 0.863852

# Collect a sample of 30 observations from the population
one_poll <- as.tbl(data.frame(vote = rbinom(30, 1, .6)))

# Resample the data using samples of size 300 (an incorrect strategy!)
one_poll_boot_300 <- one_poll %>%
  rep_sample_n(300, replace = TRUE, reps = 1000) %>%
  summarize(prop_yes_boot = mean(vote))

# Find the endpoints of the the bootstrap t-confidence interval
one_poll_boot_300 %>%
  summarize(lower = p_hat - 2 * sd(prop_yes_boot),
            upper = p_hat + 2 * sd(prop_yes_boot))
# A tibble: 1 × 2
      lower     upper
      <dbl>     <dbl>
1 0.6414326 0.7585674

# Resample the data using samples of size 3 (an incorrect strategy!)
one_poll_boot_3 <- one_poll %>%
  rep_sample_n(size = 3, replace = TRUE, reps = 1000) %>%
  summarize(prop_yes_boot = mean(vote)) 

# Find the endpoints of the the bootstrap t-confidence interval 
one_poll_boot_3 %>%
  summarize(lower = p_hat - 2 * sd(prop_yes_boot),
            upper = p_hat + 2 * sd(prop_yes_boot))
# A tibble: 1 × 2
      lower   upper
      <dbl>   <dbl>
1 0.1034601 1.29654
```

Notice how the resampled interval with size 300 was way too small and the resampled interval with size 3 was way too big.

Sample proportion value effects on bootstrap CIs
===

One additional element that changes the width of the confidence interval is the true parameter value.

When the true parameter is close to 0.5, the standard error of p̂ p^ is larger than when the true parameter is closer to 0 or 1. When calculating a bootstrap t-confidence interval, the standard error controls the width of the CI, and here the width will be narrower.

```R
# Collect 30 observations from a population with true proportion of 0.8
one_poll <- as.tbl(data.frame(vote = rbinom(n = 30, size = 1, prob = 0.8)))

# Compute p-hat of new sample: p_hat
p_hat <- mean(one_poll$vote)

# Resample the 30 observations (with replacement)
one_poll_boot <- one_poll %>%
  rep_sample_n(30, replace = TRUE, reps = 1000) %>%
  summarize(prop_yes_boot = mean(vote)) 

# Calculate the bootstrap t-confidence interval
one_poll_boot %>%
  summarize(lower = p_hat - 2 * sd(prop_yes_boot),
            upper = p_hat + 2 * sd(prop_yes_boot))
# A tibble: 1 × 2
      lower     upper
      <dbl>     <dbl>
1 0.6116044 0.9217289
```

Percentile effects on bootstrap CIs
===

Most scientists use 95% intervals to quantify their uncertainty about an estimate. That is, they understand that over a lifetime of creating confidence intervals, only 95% of them will actually contain the parameter that they set out to estimate.

There are studies, however, which warrant either stricter or more lenient confidence intervals (and subsequent error rates).

```
# Calculate a 95% bootstrap percentile interval
one_poll_boot %>% 
  summarize(q025_prop = quantile(prop_yes_boot, p = .025),
            q975_prop = quantile(prop_yes_boot, p = .975))

# Calculate a 99% bootstrap percentile interval
one_poll_boot %>% 
  summarize(q005_prop = quantile(prop_yes_boot, p = .005),
            q995_prop = quantile(prop_yes_boot, p = .995))

# Calculate a 90% bootstrap percentile interval
one_poll_boot %>% 
  summarize(q05_prop = quantile(prop_yes_boot, p = .05),
            q95_prop = quantile(prop_yes_boot, p = .95))
```

