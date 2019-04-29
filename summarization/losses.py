def wordLoss(predictions, actuals):
    wordCriterion = torch.nn.CrossEntropyLoss()
    _l = wordCriterion(predictions, actuals)
    return _l

def coverageLoss(coverages, attentions):
    total_loss = 0
    for i in range(coverages.size()[0]):
        _mins = torch.min(coverages[i], attentions[i])[0]
        _sums = torch.sum(_mins, dim=-1)
        total_loss = total_loss + _sums
    return total_loss/coverages.size()[0]