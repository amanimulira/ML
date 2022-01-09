
    from object_detection import *
    import torchvision


    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    dataset = PennFundanDataset('PennFudanPed', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,)
    # For Training
    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets) # returns losses and detections
    # for inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)

    main()