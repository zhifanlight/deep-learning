<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Solver

## 参数设置

- ```test_iter```：测试时的迭代次数

    - ```test_iter * batch_size = test_samples```

- ```lr_policy```：学习率调整策略

    - ```fixed```
        
        - 保持学习率不变
    
    - ```step```：
    
        - ```base_lr * gamma ^ [floor(iter / stepsize)]```
        
        - 需要设置 ```gamma``` 和 ```stepsize```
    
    - ```exp```：
    
        - ```base_lr * gamma ^ iter```
        
        - 需要设置 ```gamma```
        
    - ```inv```
    
        - ```base_lr * (1 + gamma * iter) ^ (-power)```
        
        - 需要设置 ```gamma``` 和 ```power```
        
    - ```multistep```
    
        - 与 ```step``` 的区别在于，只有达到 ```stepvalue``` 时才更新学习率
        
        - 需要设置多个 ```stepvalue```
        
    - ```poly```
    
        - ```base_lr * (1 - iter / max_iter) ^ power```
