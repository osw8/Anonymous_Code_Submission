









import math

def adjust_learning_rate(optimizer, epoch, args):
    """
    Linear warmup then half-cycle cosine decay.
    - Continuous at the warmup→cosine boundary.
    - Handles warmup_epochs >= epochs gracefully.
    """
    base_lr = getattr(args, "lr", None)
    min_lr  = getattr(args, "min_lr", 0.0)
    T       = getattr(args, "epochs", None)
    W       = max(0, getattr(args, "warmup_epochs", 0))

    
    if base_lr is None or T is None:
        raise ValueError("args.lr and args.epochs must be set")

    
    if T <= W or T <= 0:
        lr = base_lr * (min(epoch + 1, W) / float(max(W, 1)))  
    else:
        if epoch < W:
            
            lr = base_lr * ((epoch + 1) / float(W))
        else:
            
            denom = max(1, T - W)
            t = (min(epoch, T) - W) / float(denom)  
            lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))

    
    for pg in optimizer.param_groups:
        scale = pg.get("lr_scale", None)
        if scale is not None:
            pg["lr"] = lr * scale
        else:
            
            pg_base = pg.get("base_lr", pg.get("initial_lr", base_lr))
            if pg_base != base_lr:
                
                ratio = pg_base / float(base_lr) if base_lr > 0 else 1.0
                pg["lr"] = lr * ratio
            else:
                pg["lr"] = lr

    return lr