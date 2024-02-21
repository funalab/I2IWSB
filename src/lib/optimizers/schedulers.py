import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler_GAN(args, optimizer_G, optimizer_D):
    scheduler_G = None
    scheduler_D = None
    if hasattr(args, "lr_scheduler_G") and args.lr_scheduler is not None:
        if args.lr_scheduler == "StepLR":
            scheduler_G = lr_scheduler.StepLR(optimizer=optimizer_G,
                                              step_size=eval(args.lr_step_size),
                                              gamma=eval(args.lr_gamma)
                                              )
        elif args.lr_scheduler == "MutiStepLR":
            scheduler_G = lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                   milestones=eval(args.lr_milestones),
                                                   gamma=eval(args.lr_gamma)
                                                   )
        elif args.lr_scheduler == "ExponentialLR":
            scheduler_G = lr_scheduler.ExponentialLR(optimizer=optimizer_G,
                                                     gamma=eval(args.lr_gamma)
                                                     )
        else:
            raise ValueError("Not implemented scheduler in this version. Please check schedulers.py")

    if hasattr(args, "lr_scheduler_D") and args.lr_scheduler is not None:
        if args.lr_scheduler == "StepLR":
            scheduler_D = lr_scheduler.StepLR(optimizer=optimizer_D,
                                              step_size=eval(args.lr_step_size),
                                              gamma=eval(args.lr_gamma)
                                              )
        elif args.lr_scheduler == "MutiStepLR":
            scheduler_D = lr_scheduler.MultiStepLR(optimizer=optimizer_D,
                                                   milestones=eval(args.lr_milestones),
                                                   gamma=eval(args.lr_gamma)
                                                   )
        elif args.lr_scheduler == "ExponentialLR":
            scheduler_D = lr_scheduler.ExponentialLR(optimizer=optimizer_D,
                                                     gamma=eval(args.lr_gamma)
                                                     )
        else:
            raise ValueError("Not implemented scheduler in this version. Please check schedulers.py")

    return scheduler_G, scheduler_D

def get_scheduler(args, optimizer):
    scheduler = None
    if hasattr(args, "lr_scheduler") and args.lr_scheduler is not None:
        if args.lr_scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer=optimizer,
                                            step_size=eval(args.lr_step_size),
                                            gamma=eval(args.lr_gamma)
                                            )
        elif args.lr_scheduler == "MutiStepLR":
            scheduler = lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                 milestones=eval(args.lr_milestones),
                                                 gamma=eval(args.lr_gamma)
                                                 )
        elif args.lr_scheduler == "ExponentialLR":
            scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                   gamma=eval(args.lr_gamma)
                                                   )
        else:
            raise ValueError("Not implemented scheduler in this version. Please check schedulers.py")
    return scheduler

