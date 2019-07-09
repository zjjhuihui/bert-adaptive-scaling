# bert-adaptive-scaling
为解决目前存在的数据不均衡的问题，结合adaptive-scaling的思想，修改bert源码，提高acc&f1-score

参考的文章和github网址分别为：

参考中文文章：
A
https://mp.weixin.qq.com/s/5gxPFNxZUXAcZEGiZj1LpA

静态加权BERT：
https://blog.csdn.net/notHeadache/article/details/82183503

focal loss:
https://mp.weixin.qq.com/s?__biz=Mzg5ODAzMTkyMg==&mid=2247484118&idx=1&sn=c1f6dc300b84c067f93c261852532103&chksm=c0698a8bf71e039df1c8c2d84a43dc617d8ed26eff8ff04e728d7b1e7251e9061574f0c862d0&mpshare=1&scene=1&srcid=03146L2gE9WmuvP9vtIDy5d5&key=334230ed11c9d4e7fe6d4bd9000e662e4d040bc909b08898fdca5d173c57792e49d9ba8731f5936cddfd614dbcaef5ddb1aa9eae71445389174b1f5bcf6ef5aa14d43759508d0f5f13a828784185377a&ascene=1&uin=MTMzOTY1MzE2MQ%3D%3D&devicetype=Windows+10&version=62060833&lang=en&pass_ticket=de19jcb%2BtcTlJ9I73UhtlF9nl5ipKtmYhvUVoorlrbvPXNKFf727zzgtiQW557VN

参考论文：https://arxiv.org/pdf/1805.00250.pdf

参考github源码网址：https://github.com/sanmusunrise/AdaScaling

参考bert官网github地址：https://github.com/google-research/bert
