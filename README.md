# Predicting-Sales

1. Load the CSV file - Result is a _*Pandas*_ dataframe
   - **Input:** 
     - Critic Score (`Float` with `NaN`s)
     - Publisher (`Catagory`)
     - User Score (`Float` with `NaN`s)
     - Genre (`Catagory`)
     - Name (`String`)
     - User Count (`Float` with `NaN`s)
     - Critic Count (`Float` with `NaN`s)
   - **Output:** Global Sales (`Float`)
   
2. Tokenize Names - Bag of Words
   - The loop that builds the vocab then counts how many times that is
   ```python
   Put the Bag-of-Words code here later
   ```
   - **Result:** a list of `Int`s and how many `Int`s in that list is the size of the $n_b$ (this is vocab)
   
     - You will often see something like this - *max vocab: 200 words words(drop infrequent)* - This will keep the 200 most common words and drop the ones that don’t show up a lot
   
     > **NOTE:** Sometimes you may see an `error` that comes up that looks something like this:
     >
     > ```
     > Mat1 * Mat2, 
     > 	Shapes (1007, 2)
     > 	and (4, 97)
     > ```
     >
     > This is why the shape of you data matters 

3. Collect results from all rows (16720, $n_b$) - Save as tensor and call `t_names` 

4. Take `user_score` column 

   - Options: 
     1. Remove rows with `NaN`s - this *Pandas* function is `dropna()` 
     2. Imputation: inject mission values 
        1. Use the average value to fill in all missing values - the *Pandas* function is `fillna()` 
        2. Make it zero 
        3. Train a model on the other columns to predict the `user_score` (do not use `global_sales`) 
   - **Result:** tensor(16720, 1) 

5. Take `publisher` column - One-Hot encoding without having to tokenize it - this *Pandas* function is `get_dummies()`

   - **Result:** (16720, $n_p$) - $p$ for publisher 

6. Take `genre` 

   - **Result:** 916720, $n_g$) - $g$ for genre 

7. Figure out the total size

   ```python
   torch.cat([t_names, t_user_score, ...])
   ```

   - **Result:** (16720, $n_{v} + 1 + 1 + 1 + n_p + n_g$)

8. Build a neuron 

   ```python
   model = nn.Linear(input_dim, 1) 
   ```

   Where `input_dim` refers to the number of input features that the model, expects to receive for each individual sample in the dataset.

   