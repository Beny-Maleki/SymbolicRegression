"""# Seq2Seq-Based Symbolic Regression with Transformers

<p align="center">
  <img style="margin-top: 30px" src="https://cdn.botpenguin.com/assets/website/Sequence_to_Sequence_Models_0b5fb158a1.webp" width="500">
</p>

Symbolic regression is a method that aims to find a mathematical expression that best fits a given dataset. Unlike traditional regression techniques, symbolic regression does not assume a predefined function structure; instead, it searches for an optimal mathematical expression that relates input features to target variables. One effective approach to symbolic regression is using sequence-to-sequence (Seq2Seq) models based on transformers.

### Overview of Seq2Seq Models in Symbolic Regression:
A Seq2Seq model consists of an encoder and a decoder, both of which are typically built using transformer architectures. The main idea is to treat symbolic regression as a sequence translation problem, where:
- The input is a set of numerical data points (features).
- The output is a mathematical expression represented as a sequence of tokens.
- This approach allows the model to learn a mapping from numerical inputs to symbolic expressions, making it a powerful tool for symbolic regression.

### Encoder: Processing the Input Data

The encoder in a Seq2Seq transformer is responsible for encoding the full dataset into a latent representation. Here’s how it works:

1. Tokenization: The input dataset (typically a set of feature-value pairs) is transformed into a sequence of tokens. Each token can represent a numerical value or a special marker that indicates structural information.

2. Embedding Layer: The numerical tokens are converted into dense vector representations using an embedding layer.

3. Positional Encoding: Since transformers do not have built-in recurrence mechanisms, positional encodings are added to the embeddings to retain the order of the input sequence.

4. Self-Attention Mechanism: The encoder consists of multiple self-attention layers that allow the model to weigh the importance of different parts of the input sequence dynamically.

5. Output Representation: The encoder outputs a context-aware representation of the input sequence, which is then passed to the decoder.

### Decoder: Generating the Mathematical Expression
The decoder takes the encoder’s output and generates the target sequence, which is a symbolic mathematical expression. This process follows these steps:

1. Input Token Embedding: The decoder receives an initial input token (typically a start-of-sequence token <SOS>).

2. Self-Attention Layers: The decoder generates tokens one by one, attending to previously generated tokens using masked self-attention.

3. Cross-Attention with Encoder Outputs: The decoder also attends to the encoder’s output using cross-attention layers, ensuring that the generated expression is conditioned on the input data.

4. Token Generation via Softmax: The final layer applies a softmax function over the vocabulary (consisting of mathematical symbols, operators, and variables) to predict the next token.

In this section, we will implement symbolic regression using transformers. First, we generate the dataset and save it in the `data` folder. Do not modify the cell below. We suggest reading and understanding it first.
"""

# Don't change this cell.

C, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, x1, x2, x3, x4, x5, x6', real=True, positive=True)

MY_VOCAB = np.array([
    ['add', 4, 2],  # binary operators
    ['sub', 3, 2],
    ['mul', 6, 2],
    ['sin', 1, 1],  # unary operators
    ['cos', 1, 1],
    ['log', 2, 1],
    ['exp', 2, 1],
    ['neg', 0, 1],
    ['inv', 3, 1],
    ['sq', 2, 1],
    ['cb', 0, 1],
    ['sqrt', 2, 1],
    ['cbrt', 0, 1],
    ['C', 8, 0],  # leaves
    ['x1', 8, 0],
    ['x2', 8, 0],
    ['x3', 4, 0],
    ['x4', 4, 0],
    ['x5', 2, 0],
    ['x6', 2, 0],
])


def generate_expression(vocab):
    """
    Recursive function!
    Generate one expression using the tokens and their
    respective probabiities provided by 'vocab'.
    """
    weights = vocab[:, 1].astype('float32')
    probs = weights / np.sum(weights)
    N = len(vocab)
    expr = []
    rand_idx = np.random.choice(N, p=probs)
    cur_token = vocab[rand_idx, 0]
    cur_arity = int(vocab[rand_idx, 2])
    expr.append(cur_token)
    if cur_arity==0:
        return expr
    else:
        if cur_token in ['sin', 'cos']:
            idx1 = np.where(vocab[:, 0]=='sin')[0][0]
            idx2 = np.where(vocab[:, 0]=='cos')[0][0]
            new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
        elif cur_token in ['log', 'exp']:
            idx1 = np.where(vocab[:, 0]=='log')[0][0]
            idx2 = np.where(vocab[:, 0]=='exp')[0][0]
            new_vocab = np.delete(vocab, [idx1, idx2], axis=0)
        else:
            new_vocab = vocab
        if cur_arity==1:
            child = generate_expression(new_vocab)
            return expr + child
        elif cur_arity==2:
            child1 = generate_expression(new_vocab)
            child2 = generate_expression(new_vocab)
            return expr + child1 + child2


def translate_integers_into_tokens(seq_int):
    seq_tokens = []
    for n in range(len(seq_int)):
        if seq_int[n]>=2:
            seq_tokens.append(MY_VOCAB[seq_int[n]-2])
    return seq_tokens


def from_sequence_to_sympy(expr):
    """
    Recursive function!
    Convert the initial sequence of tokens into SymPy expression.
    """


    MY_VOCAB = np.array([
        ['add', 4, 2],  # binary operators
        ['sub', 3, 2],
        ['mul', 6, 2],
        ['sin', 1, 1],  # unary operators
        ['cos', 1, 1],
        ['log', 2, 1],
        ['exp', 2, 1],
        ['neg', 0, 1],
        ['inv', 3, 1],
        ['sq', 2, 1],
        ['cb', 0, 1],
        ['sqrt', 2, 1],
        ['cbrt', 0, 1],
        ['C', 8, 0],  # leaves
        ['x1', 8, 0],
        ['x2', 8, 0],
        ['x3', 4, 0],
        ['x4', 4, 0],
        ['x5', 2, 0],
        ['x6', 2, 0],
    ])


    cur_token = expr[0]
    try:
        return float(cur_token)  # for cases when constants are evaluated
    except ValueError:
        cur_idx = np.where(MY_VOCAB[:, 0]==cur_token)[0][0]
        cur_arity = int(MY_VOCAB[cur_idx, 2])
    if cur_arity==0:
        if cur_token=='C':
            return C
        elif cur_token=='x1':
            return x1
        elif cur_token=='x2':
            return x2
        elif cur_token=='x3':
            return x3
        elif cur_token=='x4':
            return x4
        elif cur_token=='x5':
            return x5
        elif cur_token=='x6':
            return x6
    elif cur_arity==1:
        if cur_token=='sin':
            return sympy.sin(from_sequence_to_sympy(expr[1:]))
        elif cur_token=='cos':
            return sympy.cos(from_sequence_to_sympy(expr[1:]))
        elif cur_token=='log':
            return sympy.log(from_sequence_to_sympy(expr[1:]))
        elif cur_token=='exp':
            return sympy.exp(from_sequence_to_sympy(expr[1:]))
        elif cur_token=='neg':
            return - from_sequence_to_sympy(expr[1:])
        elif cur_token=='inv':
            return 1 / from_sequence_to_sympy(expr[1:])
        elif cur_token=='sq':
            return (from_sequence_to_sympy(expr[1:]))**2
        elif cur_token=='cb':
            return (from_sequence_to_sympy(expr[1:]))**3
        elif cur_token=='sqrt':
            return sympy.sqrt(from_sequence_to_sympy(expr[1:]))
        elif cur_token=='cbrt':
            return sympy.cbrt(from_sequence_to_sympy(expr[1:]))
    elif cur_arity==2:
        arity_count = 1
        idx_split = 1
        for temp_token in expr[1:]:
            try:
                float(temp_token)  # for cases when constants are evaluated
                arity_count += -1
            except ValueError:
                temp_idx = np.where(MY_VOCAB[:, 0]==temp_token)[0][0]
                arity_count += int(MY_VOCAB[temp_idx, 2]) - 1
            idx_split += 1
            if arity_count==0:
                break
        left_list = expr[1:idx_split]
        right_list = expr[idx_split:]
        if cur_token=='add':
            return from_sequence_to_sympy(left_list) + from_sequence_to_sympy(right_list)
        elif cur_token=='sub':
            return from_sequence_to_sympy(left_list) - from_sequence_to_sympy(right_list)
        elif cur_token=='mul':
            return from_sequence_to_sympy(left_list) * from_sequence_to_sympy(right_list)


def from_sequence_to_string(expr):
    """
    OBSOLETE
    Recursive function!
    Convert the initial sequence of tokens into a string
    which can be read by SymPy.
    """


    MY_VOCAB = np.array([
        ['add', 4, 2],  # binary operators
        ['sub', 3, 2],
        ['mul', 6, 2],
        ['sin', 1, 1],  # unary operators
        ['cos', 1, 1],
        ['log', 2, 1],
        ['exp', 2, 1],
        ['neg', 0, 1],
        ['inv', 3, 1],
        ['sq', 2, 1],
        ['cb', 0, 1],
        ['sqrt', 2, 1],
        ['cbrt', 0, 1],
        ['C', 8, 0],  # leaves
        ['x1', 8, 0],
        ['x2', 8, 0],
        ['x3', 4, 0],
        ['x4', 4, 0],
        ['x5', 2, 0],
        ['x6', 2, 0],
    ])

    cur_token = expr[0]
    try:
        float(cur_token)  # for cases when constants are evaluated
        cur_arity = 0
    except ValueError:
        cur_idx = np.where(MY_VOCAB[:, 0]==cur_token)[0][0]
        cur_arity = int(MY_VOCAB[cur_idx, 2])
    if cur_arity==0:
        return cur_token
    elif cur_arity==1:
        if cur_token=='inv':
            return '1/(' + from_sequence_to_string(expr[1:]) + ')'
        elif cur_token=='sq':
            return '(' + from_sequence_to_string(expr[1:]) + ')**2'
        elif cur_token=='cb':
            return '(' + from_sequence_to_string(expr[1:]) + ')**3'
        elif cur_token=='neg':
            return '-(' + from_sequence_to_string(expr[1:]) + ')'
        else:
            return cur_token + '(' + from_sequence_to_string(expr[1:]) + ')'
    elif cur_arity==2:
        arity_count = 1
        idx_split = 1
        for temp_token in expr[1:]:
            try:
                float(temp_token)  # for cases when constants are evaluated
                arity_count += -1
            except ValueError:
                temp_idx = np.where(MY_VOCAB[:, 0]==temp_token)[0][0]
                arity_count += int(MY_VOCAB[temp_idx, 2]) - 1
            idx_split += 1
            if arity_count==0:
                break
        left_list = expr[1:idx_split]
        right_list = expr[idx_split:]
        if cur_token=='add':
            return '(' + from_sequence_to_string(left_list) + ')+(' + from_sequence_to_string(right_list) + ')'
        elif cur_token=='sub':
            return '(' + from_sequence_to_string(left_list) + ')-(' + from_sequence_to_string(right_list) + ')'
        elif cur_token=='mul':
            return '(' + from_sequence_to_string(left_list) + ')*(' + from_sequence_to_string(right_list) + ')'


def expression_tree_depth(sympy_expr):
    """
    Recursive function!
    Count the maximum depth for a given SymPy expression.
    """
    if len(sympy_expr.args)==0:
        return 1
    elif len(sympy_expr.args)==1:
        return 1 + expression_tree_depth(sympy_expr.args[0])
    else:
        max_depth = 0
        for a in sympy_expr.args:
            temp_depth = expression_tree_depth(a)
            if temp_depth > max_depth:
                max_depth = temp_depth
        return 1 + max_depth


def first_variables_first(sympy_expr):
    """
    Counts the number of variables in the SymPy expression,
    and assign firte variables first.
    Example: log(x3)+x5 becomes log(x1)+x2
    """
    tokens = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    sympy_str = str(sympy_expr)
    exist = []
    for t in tokens:
        exist.append(t in sympy_str)
    for idx_new, idx_old in enumerate(np.where(exist)[0]):
        sympy_str = sympy_str.replace(f'x{idx_old+1}', f'x{idx_new+1}')
    sympy_expr = sympy.sympify(sympy_str)
    return sympy_expr


def from_sympy_to_sequence(sympy_expr):
    """
    Recursive function!
    Convert a SymPy expression into a standardized sequence of tokens,
    which will be used as the ground truth to train the ST.
    This function calls from_sympy_power_to_sequence,
    from_sympy_multiplication_to_sequence, and
    from_sympy_addition_to sequence.
    """
    if len(sympy_expr.args)==0:  # leaf
        return [str(sympy_expr)]
    elif len(sympy_expr.args)==1:  # unary operator
        return [str(sympy_expr.func)] + from_sympy_to_sequence(sympy_expr.args[0])
    elif len(sympy_expr.args)>=2:  # binary operator
        if sympy_expr.func==sympy.core.power.Pow:
            power_seq = from_sympy_power_to_sequence(sympy_expr.args[1])
            return power_seq + from_sympy_to_sequence(sympy_expr.args[0])
        elif sympy_expr.func==sympy.core.mul.Mul:
            return from_sympy_multiplication_to_sequence(sympy_expr)
        elif sympy_expr.func==sympy.core.add.Add:
            return from_sympy_addition_to_sequence(sympy_expr)


def from_sympy_power_to_sequence(exponent):
    """
    C.f. from_sympy_to_sequence function.
    Standardize the sequence of tokens for power functions.
    """
    if exponent==(-4):
        return ['inv', 'sq', 'sq']
    elif exponent==(-3):
        return ['inv', 'cb']
    elif exponent==(-2):
        return ['inv', 'sq']
    elif exponent==(-3/2):
        return ['inv', 'cb', 'sqrt']
    elif exponent==(-1):
        return ['inv']
    elif exponent==(-1/2):
        return ['inv', 'sqrt']
    elif exponent==(-1/3):
        return ['inv', 'cbrt']
    elif exponent==(-1/4):
        return ['inv', 'sqrt', 'sqrt']
    elif exponent==(1/4):
        return ['sqrt', 'sqrt']
    elif exponent==(1/3):
        return ['cbrt']
    elif exponent==(1/2):
        return ['sqrt']
    elif exponent==(3/2):
        return ['cb', 'sqrt']
    elif exponent==(2):
        return ['sq']
    elif exponent==(3):
        return ['cb']
    elif exponent==(4):
        return ['sq', 'sq']
    else:
        return ['abort']


def from_sympy_multiplication_to_sequence(sympy_mul_expr):
    """
    C.f. from_sympy_to_sequence function.
    Standardize the sequence of tokens for multiplications.
    """
    tokens = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    nb_factors = 0
    nb_constants = 0
    is_neg = False
    for n in range(len(sympy_mul_expr.args)):
        cur_fact = sympy_mul_expr.args[n]
        if cur_fact==(-1):
            is_neg = True
        if any(t in str(cur_fact) for t in tokens):
            nb_factors += 1
        else:
            nb_constants += 1
    seq = []
    if is_neg:
        seq.append('neg')
    for _ in range(nb_factors-1):
        seq.append('mul')
    if nb_constants>0:
        seq.append('mul')
        seq.append('C')
    for n in range(len(sympy_mul_expr.args)):
        cur_fact = sympy_mul_expr.args[n]
        if any(t in str(cur_fact) for t in tokens):
            seq = seq + from_sympy_to_sequence(cur_fact)
    return seq


def from_sympy_addition_to_sequence(sympy_add_expr):
    """
    C.f. from_sympy_to_sequence function.
    Standardize the sequence of tokens for additions.
    """
    tokens = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    nb_terms = 0
    nb_constants = 0
    for n in range(len(sympy_add_expr.args)):
        cur_term = sympy_add_expr.args[n]
        if any(t in str(cur_term) for t in tokens):
            nb_terms += 1
        else:
            nb_constants += 1
    seq = []
    for _ in range(nb_terms-1):
        seq.append('add')
    if nb_constants>0:
        seq.append('add')
        seq.append('C')
    for n in range(len(sympy_add_expr.args)):
        cur_term = sympy_add_expr.args[n]
        if any(t in str(cur_term) for t in tokens):
            seq = seq + from_sympy_to_sequence(cur_term)
    return seq


def sample_from_sympy_expression(sympy_expr, nb_samples=200):
    """
    Sample from SymPy expression.
    Variables are first sampled using log-uniform distributions.
    """
    np_x = np.power(10.0, np.random.uniform(low=-1.0, high=1.0, size=(nb_samples, 6)))
    f = sympy.lambdify([x1, x2, x3, x4, x5, x6], sympy_expr)
    np_y = f(np_x[:, 0], np_x[:, 1], np_x[:, 2], np_x[:, 3], np_x[:, 4], np_x[:, 5])
    return np_y, np_x


def count_nb_variables_sympy_expr(sympy_expr):
    """
    Assumes that the variables are properly numbered, i.e.
    first_variables_first has been applied.
    Returns the number of variables in the SymPy expression.
    """
    nb_variables = 0
    while f'x{nb_variables+1}' in str(sympy_expr):
        nb_variables += 1
    return nb_variables

def is_tree_complete(seq_indices):
    """
    Check whether a given sequence of tokens defines
    a complete symbolic expression.
    """
    arity = 1
    for n in seq_indices:
        if n in [0, 1]:
            continue
            print('Predict padding or <SOS>, which is bad...')
        cur_token = MY_VOCAB[n-2]  # vocabulary is hard-coded, token 0 for padding, token 1 is <SOS>
        if cur_token in ['add', 'mul']:
            arity = arity + 2 - 1
        elif cur_token in ['sin', 'cos', 'log', 'exp', 'neg', 'inv', 'sqrt', 'sq', 'cb']:
            arity = arity + 1 - 1
        elif cur_token in ['C', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']:
            arity = arity + 0 - 1
    if arity==0:
        return True
    else:
        return False

# Don't change this cell.

# Number of initial trials
NB_TRAILS = 10000
# Minimum number of nodes (for the first crude filtering)
NB_NODES_MIN = 2
# Maximum number of nodes (for the first crude filtering)
NB_NODES_MAX = 15
# Maximum number of seconds to wait when SymPy tries to simplify
MAX_SEC_WAIT_SIMPLIFY = 5
# Minimum number of nodes (for the first crude filtering)
NB_NESTED_MAX = 6
# Minimum number of constants in the final sequences
NB_CONSTANTS_MIN = 1
# Maximum number of constants in the final sequences
NB_CONSTANTS_MAX = 1
# Maximum number of variables
NB_VARIABLES_MAX = 6
# Maximum sequence length possible (otherwise discard)
SEQ_LENGTH_MAX = 30
# Number of times we evaluate constants for each unique equation  -- BEFORE 100
NB_SAMPLING_PER_EQ = 25
# When sampling, if at least one point is above this magnitude, abort  -- BEFORE 1.0e12
ORDER_OF_MAG_LIMIT = 1.0e+9
# Number of sample points for the datasets  -- BEFORE 200
NB_SAMPLE_PTS = 50
# Variable representation: 'normal' is (y, x1, x2, ...), 'log' is log(abs(y, x1, x2, ...)), or 'both'
VARIABLE_TYPE = 'normal'
# Path output
PATH_OUT = 'data'
# Number zfill zeros
NB_ZFILL = 8

print('Generate a lot of expression trees...')
all_my_expr = []
percent = 0
for n in range(NB_TRAILS):
    if int((n+1)/NB_TRAILS*100.0) > percent:
        percent = int((n+1)/NB_TRAILS*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    cur_expr = generate_expression(MY_VOCAB)
    all_my_expr.append(cur_expr)
print(f'Nb of expression trees generated = {NB_TRAILS}')

print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print(f'Select expressions with more than {NB_NODES_MIN} and less than {NB_NODES_MAX} nodes...')
my_expr_filter = []  # Remove too simple and very long expressions
for n in range(len(all_my_expr)):
    if len(all_my_expr[n])>=NB_NODES_MIN and len(all_my_expr[n])<=NB_NODES_MAX:
        my_expr_filter.append(all_my_expr[n])
print(f'Nb of remaining expressions = {len(my_expr_filter)}')

def handler(signum, frame):
    raise Exception('too long')

print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print(f'Remove invalid or very nested (>{NB_NESTED_MAX}) expressions...')
C, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, x1, x2, x3, x4, x5, x6', real=True, positive=True)
nb_timeout_abort = 0
list_pb = []
my_expr_sympy = []
percent = 0
for n in range(len(my_expr_filter)):
    if int((n+1)/len(my_expr_filter)*100.0) > percent:
        percent = int((n+1)/len(my_expr_filter)*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    try:
        sympy_expr = from_sequence_to_sympy(my_expr_filter[n])
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(MAX_SEC_WAIT_SIMPLIFY)
        try:
            sympy_expr = sympy.factor(sympy_expr)
            sympy_expr = sympy.simplify(sympy_expr)  # so that all expressions are represented in the same way
        except Exception as e:
            nb_timeout_abort += 1
            list_pb.append(my_expr_filter[n])
            continue
        signal.alarm(0)
        if not 'zoo' in str(sympy_expr):  # only if valid expression
            if expression_tree_depth(sympy_expr) <= NB_NESTED_MAX:  # and max tree depth is not more than NB_NESTED_MAX
                sympy_expr = first_variables_first(sympy_expr)  # log(x3)+x5 becomes log(x1)+x2
                sympy_expr = sympy.factor(sympy_expr)
                sympy_expr = sympy.simplify(sympy_expr)  # so that all expressions are represented in the same way
                if 'x1' in str(sympy_expr):  # do not include if there is no variable anymore
                    my_expr_sympy.append(sympy_expr)
    except Exception as e:
        print(n, e)
        print(my_expr_filter[n])
print(f'Remaining SymPy expressions = {len(my_expr_sympy)}')
print(f'Nb aborts because timeout: {nb_timeout_abort}')

print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print('Clean the SymPy expression trees...')  # combine constants and rewrite powers/inverse/subtractions
print(f'Abort if Nb. const. < {NB_CONSTANTS_MIN} or Nb. const. > {NB_CONSTANTS_MAX}')
print(f'Abort is Nb. variables > {NB_VARIABLES_MAX}')
nb_pow_abort = 0
nb_const_min_abort = 0
nb_const_max_abort = 0
nb_var_max_abort = 0
nb_seqlen_abort = 0
my_expr_seq = []

for n in range(len(my_expr_sympy)):
    expr_seq = from_sympy_to_sequence(my_expr_sympy[n])
    if 'abort' in expr_seq:
        nb_pow_abort += 1
    else:
        if expr_seq.count('C') > NB_CONSTANTS_MAX:
            nb_const_max_abort += 1
        elif expr_seq.count('C') < NB_CONSTANTS_MIN:
            nb_const_min_abort += 1
        elif f'x{NB_VARIABLES_MAX+1}' in expr_seq:
            nb_var_max_abort += 1
        else:
            if len(expr_seq) > SEQ_LENGTH_MAX:
                nb_seqlen_abort += 1
            else:
                my_expr_seq.append(expr_seq)

print(f'Nb aborts because power exponent: {nb_pow_abort}')
print(f'Nb aborts because nb of constants: {nb_const_min_abort} and {nb_const_max_abort}')
print(f'Nb aborts because nb of variables: {nb_var_max_abort}')
print(f'Nb aborts because sequence length: {nb_seqlen_abort}')
print(f'=> Final number of expressions = {len(my_expr_seq)}')

temp = []
for n in range(len(my_expr_seq)):
    temp.append(str(my_expr_seq[n]))
temp = np.array(temp)
uniq, idx = np.unique(temp, return_index=True)

my_expr_uniq_seq = []
for n in idx:
    my_expr_uniq_seq.append(my_expr_seq[n])

print(f'\n** Number of unique expressions = {len(my_expr_uniq_seq)} **')


print('\n' + '[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')
print(f'Create {NB_SAMPLING_PER_EQ} datasets per equation.')
print(f'Datasets have {NB_SAMPLE_PTS} rows.')
print(f'Abort if generated value above {ORDER_OF_MAG_LIMIT:.1e}')
if not os.path.exists(f'{PATH_OUT}/ground_truth'):
    os.makedirs(f'{PATH_OUT}/ground_truth')
if not os.path.exists(f'{PATH_OUT}/values'):
    os.makedirs(f'{PATH_OUT}/values')

count_datasets = 0
nb_order_mag_abort = 0
nb_sample_pts_abort = 0
other_pbs_list = []
percent = 0

for n1 in range(len(my_expr_uniq_seq)):
    if int((n1+1)/len(my_expr_uniq_seq)*100.0) > percent:
        percent = int((n1+1)/len(my_expr_uniq_seq)*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    cur_seq = my_expr_uniq_seq[n1]
    try:
        for n2 in range(NB_SAMPLING_PER_EQ):
            temp = []
            cur_gt = []  # ground truth
            for n3 in range(len(cur_seq)):
                if cur_seq[n3]=='C':
                    const_val = np.round(np.random.uniform(low=-100.0, high=100.0), decimals=2)
                    temp.append(str(const_val))
                    cur_gt.append('C=' + str(const_val))
                else:
                    temp.append(cur_seq[n3])
                    cur_gt.append(cur_seq[n3])

            try:
                cur_sympy_expr = from_sequence_to_sympy(temp)
                np_y, np_x = sample_from_sympy_expression(cur_sympy_expr, nb_samples=1000)
            except Exception as e:
                other_pbs_list.append([temp, e])
                continue

            if np.nanmax(np.abs(np_y)) > ORDER_OF_MAG_LIMIT:  # if magnitude above ORDER_OF_MAG_LIMIT, abort...
                nb_order_mag_abort += 1
            else:
                if np.sum(np.logical_not(np.isnan(np_y))) < NB_SAMPLE_PTS:  # if less than 200 pts available, abort...
                    nb_sample_pts_abort += 1
                else:
                    mask = np.logical_not(np.isnan(np_y))
                    nb_temp_obs = np.sum(mask)
                    temp_np_x = np_x[mask]
                    temp_np_y = np_y[mask]
                    my_idx = np.random.choice(nb_temp_obs, size=NB_SAMPLE_PTS, replace=False)
                    nb_var = count_nb_variables_sympy_expr(cur_sympy_expr)

                    dataset = np.zeros((NB_SAMPLE_PTS, 7))
                    dataset[:, 0] = temp_np_y[my_idx]
                    dataset[:, 1:(nb_var+1)] = temp_np_x[my_idx, :nb_var]

                    np.save(f'{PATH_OUT}/values/data_{str(count_datasets).zfill(NB_ZFILL)}.npy', dataset)
                    with open(f'{PATH_OUT}/ground_truth/equation_{str(count_datasets).zfill(NB_ZFILL)}.txt', 'w') as f:
                        for token in cur_gt:
                            f.write(f'{token}\n')
                    count_datasets += 1
    except Exception as e:
        print(n1, e)
        print(cur_seq)

print(f'=> NUMBER OF DATASETS CREATED = {count_datasets}')
print('Finish!')

"""### Transformer Architecture

In this section, we will implement the Transformer architecture.
"""

import torch
import math

class TokenEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        Initializes the TokenEmbeddings module.

        Args:
            vocab_size (int): The size of the vocabulary, representing the number of unique tokens.
            d_model (int): The dimensionality of the embedding space.

        This module creates an embedding layer that maps token indices to dense vectors
        and scales the embeddings by the square root of `d_model` to maintain stable gradients.
        """

        super().__init__()

        # ============================ TODO ============================

        # ==============================================================

    def forward(self, x):
        """
        Computes the token embeddings.

        Args:
            x (Tensor): A tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            Tensor: The embedded representation of input tokens, scaled by sqrt(d_model).

        The scaling factor sqrt(d_model) helps in stabilizing gradients when used in
        Transformer models, ensuring that the variance of the input remains consistent.
        """

        # ============================ TODO ============================

        # ==============================================================

class PositionalEncodings(torch.nn.Module):
    def __init__(self, seq_length, d_model, dropout):

        """
        Initializes the PositionalEncodings module.

        Args:
            seq_length (int): The maximum sequence length for positional encoding.
            d_model (int): The dimensionality of the embeddings (must match the token embeddings).
            dropout (float): The dropout rate applied to the final positional encodings.

        This module implements sinusoidal positional encodings, which are added to token embeddings
        in Transformer models. These encodings inject information about token positions into
        the model, allowing it to capture order relationships.
        """

        # ============================ TODO ============================
        super().__init__()

        # ==============================================================

    def forward(self, x):
        """
        Computes the sinusoidal positional encodings and adds them to the input embeddings.

        Args:
            x (Tensor): A tensor of shape (batch_size, seq_length, d_model) representing token embeddings.

        Returns:
            Tensor: The input embeddings with added positional encodings, followed by dropout.

        The positional encodings use sine functions for even indices and cosine functions for odd indices,
        ensuring that each dimension has a different frequency to uniquely represent positions.
        """

        # ============================ TODO ============================

        # ==============================================================

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, h, d_model):

        """
        Initializes the MultiHeadAttention module.

        Args:
            h (int): The number of attention heads.
            d_model (int): The dimensionality of the model (must be divisible by h).

        This module implements multi-head self-attention, allowing the model to focus
        on different parts of the input sequence simultaneously. Each head independently
        computes scaled dot-product attention, and the outputs are concatenated and
        linearly transformed.
        """

        # ============================ TODO ============================
        super().__init__()

        # ==============================================================

    def forward(self, Q, K, V, mask=None):
        """
        Computes multi-head self-attention.

        Args:
            Q (Tensor): Query tensor of shape (batch_size, seq_length, d_model).
            K (Tensor): Key tensor of shape (batch_size, seq_length, d_model).
            V (Tensor): Value tensor of shape (batch_size, seq_length, d_model).
            mask (Tensor, optional): Mask tensor of shape (batch_size, 1, 1, seq_length) for attention masking.

        Returns:
            Tensor: The output of multi-head attention with shape (batch_size, seq_length, d_model).

        This function:
        1. Projects Q, K, and V using learned weight matrices.
        2. Splits the projections into multiple attention heads.
        3. Computes scaled dot-product attention for each head.
        4. Concatenates the outputs and applies a final linear transformation.
        """

        # ============================ TODO ============================

        # ==============================================================


class MLP(torch.nn.Module):
    def __init__(self, list_dims, dropout):
        """
        Initializes the Multi-Layer Perceptron (MLP) module.

        Args:
            list_dims (list of int): A list where each element represents the number of neurons
                                     in each layer, including input and output dimensions.
            dropout (float): Dropout probability applied after each layer.

        This module constructs a simple feedforward neural network with ReLU activations
        and dropout applied between layers to prevent overfitting. The last layer does
        not include an activation function.
        """

        # ============================ TODO ============================
        super().__init__()

        # ==============================================================

    def forward(self, x):

        """
        Computes the forward pass of the MLP.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: The output of the MLP after passing through all layers.

        The input is sequentially passed through linear layers, ReLU activations,
        and dropout layers (if applicable). The last layer does not apply ReLU or dropout.
        """

        # ============================ TODO ============================

        # ==============================================================

class EncoderLayerMix(torch.nn.Module):


    def __init__(self, nb_samples, max_nb_var, d_model, h, dropout):

        """
        Initializes the EncoderLayerMix module.

        Args:
            nb_samples (int): The number of samples in the input batch.
            max_nb_var (int): The maximum number of variables in the input sequence.
            d_model (int): The dimensionality of the model.
            h (int): The number of attention heads in multi-head attention.
            dropout (float): Dropout probability applied to regularize the model.

        This module represents an encoder layer that combines an MLP-based transformation
        with multi-head attention and a residual connection followed by layer normalization.
        The MLP compresses the input before applying attention, making it suitable for
        structured data.

        """

        # ============================ TODO ============================
        super().__init__()
        # ==============================================================

    def forward(self, x):

        """
        Computes the forward pass of the EncoderLayerMix.

        Args:
            x (Tensor): Input tensor of shape (batch_size, nb_samples, max_nb_var, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, nb_samples, max_nb_var, d_model).

        The forward pass consists of:
        1. Flattening the input over the variable dimension.
        2. Applying an MLP to reduce dimensionality.
        3. Using multi-head self-attention to refine representations.
        4. Applying dropout to the attention output.
        5. Using an Add & Norm layer for residual learning.
        """

        # ============================ TODO ============================

        # ==============================================================

class Encoder(torch.nn.Module):
        def __init__(self, nb_samples, max_nb_var, d_model, h, N, dropout):
            """
            Initializes the Encoder module.

            Args:
                nb_samples (int): The number of samples in the input batch.
                max_nb_var (int): The maximum number of variables in the input sequence.
                d_model (int): The dimensionality of the model.
                h (int): The number of attention heads in multi-head attention.
                N (int): The number of stacked encoder layers.
                dropout (float): Dropout probability applied throughout the model.

            This module represents a Transformer-style encoder composed of `N` stacked `EncoderLayerMix` layers.
            After the encoding process, an additional MLP is applied, followed by max pooling
            to make the representation permutation-invariant with respect to sample points.


            The `Encoder` module is a Transformer-style encoder designed to process an input sequence while ensuring
            permutation invariance at the sample level. It consists of three key components:

            1. First MLP (`first_mlp`)
            2. Stacked Encoder Layers (`layers`)
            3. Final MLP (`last_mlp`)

            """

            # ============================ TODO ============================
            super().__init__()

            # ==============================================================

        def forward(self, x):

            """
            Performs a forward pass through the Encoder.

            Args:
                x (Tensor): Input tensor of shape (batch_size, nb_samples, max_nb_var, d_model).

            Returns:
                Tensor: Encoded representation of shape (batch_size, d_model).

            The forward pass consists of:
            1. Passing the input through `N` encoder layers.
            2. Applying a final MLP transformation.
            3. Performing max pooling along the sample dimension to make the output
            permutation-invariant with respect to sample points.
            """

            # ============================ TODO ============================

            # ==============================================================

class AddAndNorm(torch.nn.Module):
    def __init__(self, d_model):
        """
        Initializes the AddAndNorm module.

        Parameters:
        d_model (int): The number of expected features in the input tensor.

        Components:
        - LayerNorm: Applies layer normalization to stabilize training and improve convergence.
        """

        # ============================ TODO ============================
        super().__init__()
        # ==============================================================

    def forward(self, x_input, x_output):
        """
        Forward pass of the AddAndNorm module.

        Parameters:
        x_input (torch.Tensor): The original input tensor (residual connection).
        x_output (torch.Tensor): The output tensor from a sublayer (e.g., feed-forward or attention).

        Returns:
        torch.Tensor: The normalized tensor after applying residual connection and layer normalization.

        Process:
        - Adds the input tensor (`x_input`) and output tensor (`x_output`).
        - Applies layer normalization to the sum.
        """

        # ============================ TODO ============================

        # ==============================================================


class DecoderLayer(torch.nn.Module):
    def __init__(self, h, d_model, dropout):

        """
        Initializes the DecoderLayer.

        Parameters:
        h (int): The number of attention heads in multi-head attention.
        d_model (int): The dimension of model embeddings.
        dropout (float): Dropout probability to prevent overfitting.

        Components:
        - MultiHeadAttention (self-attention for decoder input).
        - Dropout layer after self-attention.
        - AddAndNorm layer for residual connection and normalization.
        - MultiHeadAttention (encoder-decoder attention).
        - Dropout layer after encoder-decoder attention.
        - AddAndNorm layer for second residual connection.
        - MLP (Feedforward neural network).
        - AddAndNorm layer for the final residual connection.
        """

        # ============================ TODO ============================
        super().__init__()
        # ==============================================================

    def forward(self, input_dec, mask_dec, output_enc):
        """
        Forward pass of the Transformer Decoder Layer.

        Parameters:
        input_dec (torch.Tensor): Decoder input tensor (sequence embedding).
        mask_dec (torch.Tensor): Mask tensor to prevent attending to future tokens.
        output_enc (torch.Tensor): Encoder output tensor (used in encoder-decoder attention).

        Returns:
        torch.Tensor: Output tensor after applying self-attention, encoder-decoder attention,
                      and feedforward network with residual connections.

        Process:
        1. **Self-Attention:** Applies self-attention on `input_dec` with a causal mask.
        2. **Dropout & Residual Connection:** Applies dropout and adds the residual connection.
        3. **Layer Normalization:** Normalizes the output of the residual connection.
        4. **Encoder-Decoder Attention:** Allows decoder to attend to encoder outputs.
        5. **Dropout & Residual Connection:** Applies dropout and adds the residual connection.
        6. **Layer Normalization:** Normalizes the output of encoder-decoder attention.
        7. **Feedforward Network (MLP):** Applies a position-wise feedforward network.
        8. **Final Residual Connection & Normalization:** Normalizes the final output.

        """

        # ============================ TODO ============================

        # ==============================================================

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, seq_length, d_model, h, N, dropout):

        """
        Initializes the Transformer Decoder.

        Parameters:
        vocab_size (int): Number of unique tokens in the vocabulary.
        seq_length (int): Maximum length of input sequences.
        d_model (int): Dimensionality of embedding vectors.
        h (int): Number of attention heads in multi-head attention.
        N (int): Number of decoder layers.
        dropout (float): Dropout probability to prevent overfitting.

        Components:
        - TokenEmbeddings: Converts token indices to dense embedding vectors.
        - PositionalEncodings: Adds positional information to embeddings.
        - Dropout: Applies dropout for regularization.
        - DecoderLayer(s): A stack of `N` Transformer decoder layers.
        """

        # ============================ TODO ============================
        super().__init__()
        # ==============================================================

    def forward(self, target_seq, mask_dec, output_enc):

        """
        Forward pass of the Transformer Decoder.

        Parameters:
        target_seq (torch.Tensor): The input target sequence (batch_size, seq_length).
        mask_dec (torch.Tensor): Decoder mask to prevent attending to future tokens.
        output_enc (torch.Tensor): The encoded output from the encoder.

        Returns:
        torch.Tensor: The final decoder output tensor after passing through all layers.

        Process:
        1. **Token Embedding:** Convert input tokens to dense vectors.
        2. **Positional Encoding:** Inject sequence position information.
        3. **Dropout:** Apply dropout for regularization.
        4. **Pass Through Decoder Layers:** Each layer applies self-attention, encoder-decoder attention, and feedforward processing.
        """

        # ============================ TODO ============================

        # ==============================================================


class TransformerModel(torch.nn.Module):
    def __init__(self, nb_samples, max_nb_var, d_model, vocab_size, seq_length, h, N_enc, N_dec, dropout):

        """
        A full Transformer model composed of an Encoder and a Decoder.
        This model is used for sequence-to-sequence tasks, such as symbolic regression or machine translation.

        Args:
            nb_samples (int): Number of input samples.
            max_nb_var (int): Maximum number of variables in input samples.
            d_model (int): Dimensionality of model embeddings.
            vocab_size (int): Size of the vocabulary.
            seq_length (int): Maximum length of target sequences.
            h (int): Number of attention heads.
            N_enc (int): Number of encoder layers.
            N_dec (int): Number of decoder layers.
            dropout (float): Dropout rate for regularization.
        """


        # ============================ TODO ============================

        super().__init__()

        # ==============================================================


    def forward(self, input_enc, target_seq):
        """
        Forward pass of the TransformerModel.

        Parameters:
        input_enc (torch.Tensor): The input tensor to the encoder (batch_size, nb_samples, max_nb_var).
        target_seq (torch.Tensor): The target sequence tensor for decoding (batch_size, seq_length).

        Returns:
        torch.Tensor: Final output logits (batch_size, seq_length, vocab_size), ready for softmax activation.

        Process:
        1. **Masking:**
           - Creates a padding mask to ignore padding tokens.
           - Generates a future mask to prevent attending to future tokens during decoding.
           - Combines both masks for masked self-attention in the decoder.
        2. **Encoder Processing:**
           - Encodes the input sequence into a high-level representation.
        3. **Decoder Processing:**
           - Uses the encoded representation and the masked target sequence to generate predictions.
        4. **Final Projection:**
           - Passes the decoder output through a linear layer to obtain vocabulary logits.
        """

        # ============================ TODO ============================

        # ==============================================================

def compute_transformer_loss(prediction, target, label_smooth=0.0):
    """
    **TASK: Implement the loss function for a Transformer model.**

    Hints:
    - Use `torch.nn.CrossEntropyLoss` for computing loss.
    - **Ignore padding tokens (index `0`)** in loss computation to avoid penalizing padding.
    - Apply **label smoothing** (if `label_smooth > 0.0`) to regularize training and prevent overfitting.

    Parameters:
    - `prediction`: A tensor of shape (batch_size, seq_length, vocab_size) representing the model's output logits.
    - `target`: A tensor of shape (batch_size, seq_length) containing the ground truth token indices.
    - `label_smooth`: A floating-point value for label smoothing (default is `0.0`, meaning no smoothing).

    Steps to Implement:
    1. Create a `CrossEntropyLoss` instance.
       - Set `ignore_index=0` to **exclude padding tokens** from loss computation.
       - Use `label_smoothing=label_smooth` for better generalization.
    2. Compute the loss by passing `prediction` and `target` to the loss function.
    3. Return the computed loss.
    """

    # ============================= TODO =============================

    # ================================================================



def compute_transformer_accuracy(prediction, target):

    """
    **TASK: Implement the accuracy computation for a Transformer model.**

    Hints:
    - Accuracy is measured by comparing the model's predicted token indices with the actual target token indices.
    - **Ignore padding tokens (index `0`)** when computing accuracy, as they do not contribute to meaningful evaluation.
    - Use `torch.argmax(prediction, dim=-1)` to get the most likely token for each position.

    Parameters:
    - `prediction`: A tensor of shape (batch_size, seq_length, vocab_size) representing the model's output logits.
    - `target`: A tensor of shape (batch_size, seq_length) containing the ground truth token indices.

    Steps to Implement:
    1. **Create a mask for padding tokens**:
       - Identify positions where `target == 0` (padding).
    2. **Compute accuracy for non-padding tokens**:
       - Get the predicted tokens using `torch.argmax(prediction, dim=-1)`.
       - Compare them with `target` using `torch.eq()`.
       - Ensure padding positions are ignored using `torch.logical_and()`.
    3. **Compute the final accuracy**:
       - Count the number of correct predictions (`torch.sum(correct_bool)`).
       - Normalize by the total number of non-padding tokens.

    """

    # ============================= TODO =============================

    # ================================================================

MY_VOCAB = [
    'add',  # 2
    'mul',  # 3
    'sin',  # 4
    'cos',  # 5
    'log',  # 6
    'exp',  # 7
    'neg',  # 8
    'inv',  # 9
    'sqrt',  # 10
    'sq',  # 11
    'cb',  # 12
    'C',  # 13
    'x1',  # 14
    'x2',  # 15
    'x3',  # 16
    'x4',  # 17
    'x5',  # 18
    'x6',  # 19
]


"""
Fill Hyper-Parameters here.
"""

PATH_DATA = './data'
NB_ZFILL = 8

NB_EPOCHS =          #TODO
BATCH_SIZE =         #TODO
TRAIN_PROP =         #TODO
VAL_PROP =           #TODO

D_MODEL =            #TODO
H =                    #TODO
N_ENC =                #TODO
N_DEC =                #TODO
DROPOUT =           #TODO

"""### Load Data:
This cell is responsible for loading the data. torch_inputs represents the model inputs and the dataset, which has the shape `(total_dataset_size, number_of_samples, max_number_of_variables)`, specifically `(8239, 50, 7)`. torch_targets contains the correct expressions, with a shape of `(total_dataset_size, max_length_expression)`, which is `(8239, 20)`.

You only need to split the dataset into training, validation, and test sets in the correct order. Remember to shuffle the data before splitting.
"""

nb_data = len(glob.glob(f'{PATH_DATA}/values/data_*.npy'))
print(f'\nNb datasets = {nb_data}')

data_values = []
data_tokens = []
percent = 0
for n in range(nb_data):
    if int((n + 1) / nb_data * 100.0) > percent:
        percent = int((n+1)/nb_data*100.0)
        print(f'{percent}% ', end='', flush=True)
        if percent%10==0:
            print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', flush=True)
    cur_path = f'{PATH_DATA}/values/data_{str(n).zfill(NB_ZFILL)}.npy'
    data_values.append(np.load(cur_path))
    cur_path = f'{PATH_DATA}/ground_truth/equation_{str(n).zfill(NB_ZFILL)}.txt'
    with open(cur_path) as f:
        lines = []
        for token in f.readlines():
            assert token[-1]=='\n'
            if token[0]=='C':
                lines.append('C')
            else:
                lines.append(token[:-1])
    data_tokens.append(lines)

data_values = np.array(data_values)
print(f'Shape of all datasets = {data_values.shape}')
print(f'Lenght of ground truth = {len(data_tokens)}')

max_seq_length = 0
for n in range(nb_data):
    if len(data_tokens[n])>max_seq_length:
        max_seq_length = len(data_tokens[n])
print(f'\nMax sequence length = {max_seq_length}')

vocab_size = len(MY_VOCAB)
print('Vocabulary:')
print(MY_VOCAB)
print(f'Vocab size = {vocab_size}')

data_targets = np.zeros((nb_data, max_seq_length + 1))  # <SOS> until max_seq_length
for n1 in range(nb_data):
    data_targets[n1, 0] = 1  # 1 is <SOS>
    for n2 in range(len(data_tokens[n1])):
        data_targets[n1, n2 + 1] = MY_VOCAB.index(data_tokens[n1][n2]) + 2  # from 2 to vocab_size + 2

# Transform data into Torch tensors
torch_inputs = torch.from_numpy(data_values).unsqueeze(-1).type(torch.float32)
nb_samples = torch_inputs.shape[1]
torch_targets = torch.from_numpy(data_targets).type(torch.FloatTensor).type(torch.int64)
print(f'Dataset input shape = {torch_inputs.shape}')
print(f'Dataset target shape = {torch_targets.shape}')
print(f'Nb samples = {nb_samples}')

# Split into {train, validation, test} sets with correct proportions
# ============================= TODO =============================
nb_obs = torch_inputs.shape[0]
train_idx =  # TODO
val_idx =  # TODO
test_idx =  # TODO
# ================================================================

nb_train_obs = len(train_idx)
nb_val_obs = len(val_idx)
nb_test_obs = len(test_idx)
nb_train_step_per_epoch = math.ceil(nb_train_obs / BATCH_SIZE)
nb_val_step_per_epoch = math.ceil(nb_val_obs / BATCH_SIZE)
nb_test_step = math.ceil(nb_test_obs / BATCH_SIZE)
print(f'Batch size = {BATCH_SIZE}')
print(f'Nb training steps per epoch = {nb_train_step_per_epoch}')
print(f'Nb val steps per epoch = {nb_val_step_per_epoch}')
print(f'Nb final test steps = {nb_test_step}')

# Init Model and Optimizer for training

# ========================== TODO ==========================

# ==========================================================

def training_step(trainX, trainY, target):

    """
    **TASK: Implement a single training step for the Transformer model.**

    Parameters:
    - `trainX` (torch.Tensor): Input tensor for the encoder.
    - `trainY` (torch.Tensor): Input tensor for the decoder (teacher forcing).
    - `target` (torch.Tensor): Ground truth target tensor.

    Returns:
    - `loss` (torch.Tensor): Computed loss value.
    - `acc` (torch.Tensor): Computed accuracy.

    """

    # ==================================== TODO ====================================

    # ==============================================================================

def validation_step(valX, valY, target):
    """
    **TASK: Implement a single validation step for the Transformer model.**

    Parameters:
    - `valX` (torch.Tensor): Input tensor for the encoder.
    - `valY` (torch.Tensor): Input tensor for the decoder (teacher forcing).
    - `target` (torch.Tensor): Ground truth target tensor.

    Returns:
    - `loss` (torch.Tensor): Computed loss value.
    - `acc` (torch.Tensor): Computed accuracy.

    """

    # ==================================== TODO ====================================

    # ==============================================================================

"""
    **TASK: Implement the full training and validation loop for the Transformer model.**

    Hints:
    - Each epoch consists of a **training phase** and a **validation phase**.
    - **Training Phase:**
      1. Shuffle the training indices (`train_idx`) to ensure randomness.
      2. Iterate through mini-batches of training data.
      3. Extract batch data using the following method:
         ```
         trainX = torch_inputs[batch_idx]
         trainY = torch_targets[batch_idx, :-1]  # Decoder input (shifted left)
         target = torch_targets[batch_idx, 1:]  # Target output (shifted right)
         ```

    - **Validation Phase:**
      1. Shuffle validation indices (`val_idx`) for randomness.

    **NOTE:** The decoder output is **shifted by one position** for proper training.
"""


# ======================================== TODO ========================================

# ======================================================================================

"""## Test
After training, we need to evaluate the model on our datasets. First, create a dataset using the first given equation and compare it with the model's results. Then, repeat this process for the second dataset. Make sure to plot both the true dataset and the expression predicted by the model for visualization.
To test the model, simply input the dataset into the `evaluate` function. For example, if your dataset consists of two variables, `x_1` and `x_2`, along with the output `y`, you can structure it as follows:

```
dataset[:, 0] = y_values
dataset[:, 1] = x1_values
dataset[:, 2] = x2_values
```

Then, pass this dataset to the model. The evaluate function returns two outputs:
    1. sympy_pred: The predicted equation in SymPy format.
    2. string_pred: A string representation of the equation that you can use for plotting the results.

You can use string_pred as a function like this:

```
result = eval(string_pred, {"C": 5, "x1": 2, "x2": 5, "log": np.log})
```

Additionally, optimize parameters to find the best values for each variable. Once optimized, plot the best results for a clearer comparison.
"""

def evaluate(dataset):

    encoder_input = torch.Tensor(dataset).unsqueeze(0).unsqueeze(-1)
    encoder_input = encoder_input.to(device)
    dataset = encoder_input
    encoder_output = transformer.encoder(dataset)  # Encoder output is fixed for the batch

    seq_length = transformer.decoder.positional_encoding.seq_length
    decoder_output = torch.zeros((dataset.shape[0], seq_length + 1), dtype=torch.int64)  # initialize Decoder output
    decoder_output[:, 0] = 1
    is_complete = torch.zeros(dataset.shape[0], dtype=torch.bool).to(device)  # check when decoding is finished
    decoder_output = decoder_output.to(device)
    for n1 in range(seq_length):
        padding_mask = torch.eq(decoder_output[:, :-1], 0).unsqueeze(1).unsqueeze(1).to(device)
        future_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(device)
        mask_dec = torch.logical_or(padding_mask, future_mask)
        temp = transformer.decoder(
            target_seq=decoder_output[:, :-1],
            mask_dec=mask_dec,
            output_enc=encoder_output,
        )
        temp = transformer.last_layer(temp)

        decoder_output[:, n1+1] = torch.where(is_complete, 0, torch.argmax(temp[:, n1], axis=-1))
        for n2 in range(dataset.shape[0]):
            if is_tree_complete(decoder_output[n2, 1:]):
                is_complete[n2] = True
    decoder_tokens = translate_integers_into_tokens(decoder_output[0])
    sympy_pred = from_sequence_to_sympy(decoder_tokens)
    string_pred = from_sequence_to_string(decoder_tokens)
    return sympy_pred, string_pred

# First Dataset

C, y, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, y, x1, x2, x3, x4, x5, x6', real=True, positive=True)


dataset = np.zeros((50, 7))

dataset[:, 0] =  #TODO
dataset[:, 1] =  #TODO

# use evaluate function to find the expression
sympy_pred, string_pred = evaluate(dataset)
print(string_pred)
plot_symbolic(x1_values, y_values, string_pred.replace("C", "-1"))

# Second Dataset

C, y, x1, x2, x3, x4, x5, x6 = sympy.symbols('C, y, x1, x2, x3, x4, x5, x6', real=True, positive=True)

# Make tabular dataset

dataset[:, 0] = #TODO
dataset[:, 1] =  #TODO
dataset[:, 2] =  #TODO

# use evaluate function to find the expression
sympy_pred, string_pred = evaluate(dataset)
print(string_pred)

# Parameter optimization using mse loss

# Plot the results on two datasets.
