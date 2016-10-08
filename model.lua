require("hdf5")
require("nn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-eta', '0.0001', 'learning rate')
cmd:option('-nepochs', '20', 'number of epochs')
cmd:option('-dp', '10', 'dp size')

function main()
   -- Parse input parameters
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nepochs = tonumber(opt.nepochs)
   eta = tonumber(opt.eta)
   dp_size = tonumber(opt.dp)
   big_matrix_train_in = f:read('big_matrix_train_in'):all():double()
   big_matrix_test_in = f:read('big_matrix_test_in'):all():double()
   big_matrix_train_out = f:read('big_matrix_train_out'):all():double()
   big_matrix_test_out = f:read('big_matrix_test_out'):all():double()
   embedding_matrix = f:read('embedding_matrix'):all():double():transpose(1, 2)
   num_bills = f:read('num_bills'):all():long()[1]
   num_words = big_matrix_train_in[1]:size()[2]
   word_embed_len = 50
   num_cp = f:read('num_cp'):all():long()[1]
   -- Models
   print("Number of bills: ", num_bills)
   print("Baseline accuracy: ", get_baseline(big_matrix_train_out[1], big_matrix_test_out[1]))
   local start_time = os.time()
   -- Run model using cross-validation for replication purposes
   if opt.classifier == 'nn_embed_m' then
     for i = 1, big_matrix_train_in:size()[1] do
       doc_term_matrix_train = big_matrix_train_in[i]
       doc_term_matrix_test = big_matrix_test_in[i]
       vote_matrix_train = big_matrix_train_out[i]
       vote_matrix_test = big_matrix_test_out[i]
       local nn_model = train_nn_embed_m(doc_term_matrix_train, vote_matrix_train, false)
       print("Train time (seconds):", os.time()-start_time)
       local mod_test = make_sparse_list_input(doc_term_matrix_test)
       local accuracy = nn_get_acc_m(nn_model, mod_test, vote_matrix_test)
       print("Test accuracy:", accuracy)
     end
   end
   -- Run model without cross-validation for data generation purposes
   if opt.classifier == 'nn_embed_m_nocv' then
     doc_term_matrix_train = big_matrix_train_in[1]
     doc_term_matrix_test = big_matrix_test_in[1]
     vote_matrix_train = big_matrix_train_out[1]
     vote_matrix_test = big_matrix_test_out[1]
     local nn_model = train_nn_embed_m(doc_term_matrix_train, vote_matrix_train, true)
     print("Train time (seconds):", os.time()-start_time)
     local mod_test = make_sparse_list_input(doc_term_matrix_test)
     local accuracy = nn_get_acc_m(nn_model, mod_test, vote_matrix_test)
     print("Test accuracy:", accuracy)
  end
end

-- Process input for model
function make_sparse_list_input(inp)
  retset = {}
  for i = 1, inp:size()[1] do
    vec_len = inp[i]:sum(1)[1]
    vec = torch.zeros(vec_len)
    k = 1
    for j = 1, inp:size()[2] do
      if inp[i][j] > 0 then
        vec[k] = j
        k = k + 1
      end
    end
    retset[i] = vec
  end
  return retset
end

-- Run main model
function train_nn_embed_m(inp, out, nocv)
  local model = nn.Sequential()

  local proc_bill = nn.Sequential()
  proc_bill:add(nn.LookupTable(num_words, word_embed_len))
  proc_bill:add(nn.Mean(1, 2))
  proc_bill:add(nn.Linear(word_embed_len, dp_size))

  proc_bill.modules[1].weight = embedding_matrix:transpose(1, 2)

  for i = 1, proc_bill.modules[3].weight:size()[1] do
    for j = 1, proc_bill.modules[3].weight:size()[2] do
      proc_bill.modules[3].weight[i][j] = torch.uniform(-0.01, 0.01)
    end
  end

  local proc_cp = nn.Sequential()
  proc_cp:add(nn.LookupTable(num_cp,dp_size))

  for i = 1, proc_cp.modules[1].weight:size()[1] do
    for j = 1, proc_cp.modules[1].weight:size()[2] do
      proc_cp.modules[1].weight[i][j] = torch.uniform(-0.01, 0.01)
    end
  end

  local par_model = nn.ParallelTable()
  par_model:add(proc_bill)
  par_model:add(proc_cp)

  model:add(par_model)
  model:add(nn.DotProduct())

  model:add(nn.Sigmoid())

  nll = nn.BCECriterion()

  local_params, grad_params = model:getParameters()

  inp = make_sparse_list_input(inp)
  local mod_test = make_sparse_list_input(doc_term_matrix_test)

  for ep = 1, nepochs do
    print("Epoch: ", ep)
    acc_train = nn_get_acc_m(model, inp, out)
    print("Start-of-epoch train accuracy: ", acc_train)
    acc_test = nn_get_acc_m(model, mod_test, vote_matrix_test)
    print("Start-of-epoch test accuracy: ", acc_test)
    prec, recall = nn_get_prec_rec_m(model, mod_test, vote_matrix_test)
    print ("Precision, Recall, F1:", prec, recall, 2 * prec * recall / (prec + recall))
    for i = 1, out:size()[1] do
      for j = 1, out:size()[2] do
        if out[i][j] > 1 then
          local X = inp[i]
          local c = torch.ones(1) * j
          local y = torch.ones(1) * (out[i][j] - 2)
          model:zeroGradParameters()
          local pred = model:forward({X, c})
          local err = nll:forward(pred, y)
          local t = nll:backward(pred, y)
          model:backward({X, c}, t)
          grad_params:view(1,grad_params:size()[1]):renorm(2,1,1.0)
          model:updateParameters(eta)
        end
      end
    end
  end
  if (nocv == true) then
    io.output("cp_weights.txt")
    for i = 1, proc_cp.modules[1].weight:size()[1] do
      for j = 1, proc_cp.modules[1].weight:size()[2] do
        io.write(proc_cp.modules[1].weight[i][j])
        io.write(" ")
      end
      io.write("\n")
    end
    io.output("bill_weights.txt")
    bill_weights = proc_bill.modules[1].weight * proc_bill.modules[3].weight:transpose(1, 2)
    for i = 1, bill_weights:size()[1] do
      for j = 1, bill_weights:size()[2] do
        io.write(bill_weights[i][j])
        io.write(" ")
      end
      io.write("\n")
    end
    io.output(io.stdout)
  end
  return model
end

-- Get accuracy of model given model and test set
function nn_get_acc_m(model, inp, out)
  local num_count = 0.0
  local denom_count = 0.0
  for i = 1, out:size()[1] do
    for j = 1, out:size()[2] do
      if out[i][j] > 1 then
        local X = inp[i]
        local c = torch.ones(1) * j
        local y = torch.ones(1) * (out[i][j] - 2)
        local pred = model:forward({X, c})
        if pred[1] >= 0.5 then
          pred = 1
        else
          pred = 0
        end
        if pred == y[1] then
          num_count = num_count + 1
        end
        denom_count = denom_count + 1
      end
    end
  end
  return num_count/denom_count
end

-- Get precision and recall of model given model and test set
function nn_get_prec_rec_m(model, inp, out)
  local true_positives = 0.0
  local positives = 0.0
  local trues = 0.0
  for i = 1, out:size()[1] do
    for j = 1, out:size()[2] do
      if out[i][j] > 1 then
        local X = inp[i]
        local c = torch.ones(1) * j
        local y = torch.ones(1) * (out[i][j] - 2)
        local pred = model:forward({X, c})
        if pred[1] >= 0.5 then
          pred = 1
        else
          pred = 0
        end
        if y[1] == 0 and pred == 0 then
          true_positives = true_positives + 1
        end
        if y[1] == 0 then
          trues = trues + 1
        end
        if pred == 0 then
          positives = positives + 1
        end
      end
    end
  end
  return true_positives/positives, true_positives/trues
end

-- Calculate baseline accuracy for a congress given test and training sets
function get_baseline(out1, out2)
  local num_count = 0
  local denom_count = 0
  for i = 1, out1:size()[1] do
    for j = 1, out1:size()[2] do
      if out1[i][j] > 1 then
        if out1[i][j] - 2 == 1 then
          num_count = num_count + 1
        end
        denom_count = denom_count + 1
      end
    end
  end
  for i = 1, out2:size()[1] do
    for j = 1, out2:size()[2] do
      if out2[i][j] > 1 then
        if out2[i][j] - 2 == 1 then
          num_count = num_count + 1
        end
        denom_count = denom_count + 1
      end
    end
  end
  return num_count/denom_count
end

main()
