function dist = testModel(prediction,label,target)
    dist = [];
    dist(1) = AveragePrecision(prediction',target');
    dist(2) = Coverage(prediction',target');
    dist(3) = OneError(prediction',target');
    dist(4) = RankingLoss(prediction',target');
    dist(5) = HammingLoss(label',target');