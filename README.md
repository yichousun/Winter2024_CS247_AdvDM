# Winter 2024 CS 247: Advanced Data Mining
### Instructor: Yizhou Sun
- Lecture Time: Monday/Wednesday 10-11:50pm
-	Classroom: BH 4760
-	Office hours: Tuesday 4:45-6:15pm @ BH 3531F


### TA:
-	Fred Xu (fredxu at cs.ucla.edu), office hours: Monday 4-5PM and Wednesday 4-5PM
-	Yanqiao Zhu (yzhu at cs.ucla.edu), office hours: Monday 8-9PM (Zoom) and Tuesday in person 10-11am
-	TA Office: BH 3256S


## Course Description
This course introduces concepts, algorithms, and techniques of data mining on different types of datasets, which covers basic data mining algorithms, as well as advanced topics on text mining, graph/network mining, and recommender systems. A team-based course project involving hands-on practice of mining useful knowledge from large data sets is required, in addition to regular assignments. The course is a graduate-level computer science course, which is also a good option for senior undergraduate students who are interested in the field as well as students from other disciplines who need to understand, develop, and use data mining systems to analyze large amounts of data.

## Prerequisites
- Required prerequisite courses: CS 145 or CS 146 or equivalent. 
- You are expected to have background knowledge in data structures, algorithms, basic linear algebra, and basic statistics. 
- You are expected to know basic knowledge in data mining and machine learning. 
- You will also need to be familiar with at least one programming language (Python will be used for homework), and have programming experiences. 

## Learning Objectives
- Review and understand fundamentals of basic data mining techniques
- Learn recent data mining techniques on several advanced data types
- Develop skills to apply data mining algorithms to solve real-world applications
- Gain initial experience in conducting research on data mining

## Grading
-	Homework: 40%
-	Course project: 30%
-	Exam: 25%
-	Participation: 5%


*All the deadlines are 11:59PM (midnight) of the due dates.

*Late submission policy: you will get original score * <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbf{1}(t<=24)e^{-(ln(2)/12)*t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbf{1}(t<=24)e^{-(ln(2)/12)*t}" title="\mathbf{1}(t<=24)e^{-(ln(2)/12)*t}" /></a>, if you are t hours late.

*No copying or sharing of homework!

- You can discuss general challenges and ideas with others.
- Suspicious cases will be reported to The Office of the Dean of Students.

## Q & A
-	We will be using [Piazza](piazza.com/ucla/winter2024/cs247) for class discussion. The system is highly catered to getting you help fast and efficiently from classmates, the TAs, and myself. Rather than emailing questions to the teaching staff, I encourage you to post your questions on Piazza.
-	Sign up Piazza here: [piazza.com/ucla/winter2024/cs247](piazza.com/ucla/winter2024/cs247)
-	Tips: Answering other students' questions will increase your participation score.

## Academic Integrity Policy
"With its status as a world-class research institution, it is critical that the University uphold the highest standards of integrity both inside and outside the classroom. As a student and member of the UCLA community, you are expected to demonstrate integrity in all of your academic endeavors. Accordingly, when accusations of academic dishonesty occur, The Office of the Dean of Students is charged with investigating and adjudicating suspected violations. Academic dishonesty, includes, but is not limited to, cheating, fabrication, plagiarism, multiple submissions or facilitating academic misconduct."
For more information, please refer to the <a href="https://www.deanofstudents.ucla.edu/portals/16/documents/studentguide.pdf"> guidance </a>.

## Tentative Schedule
| Week | Date | Topic | Further Reading | Discussion Session| Homework| Course Project|
| ------- | ------ | ------ | -------- | ------ | ------ | ------ |
| Week 1 | 1/8 | Introduction [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/01Intro.pdf); Basics: Naive Bayes [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/02NaiveBayes_LR.pdf) | <ul><li>[Review of probability from a course by David Blei](http://www.cs.princeton.edu/courses/archive/spring07/cos424/scribe_notes/0208.pdf) from Princeton U.</li><li>[Machine Learning Math Essentials](http://courses.washington.edu/css490/2012.Winter/lecture_slides/02_math_essentials.pdf) by Jeff Howbert from Washington U.</li><li>[http://cs229.stanford.edu/section/cs229-prob.pdf](http://cs229.stanford.edu/section/cs229-prob.pdf)</li><li>[Optimization](http://web.cs.ucla.edu/~yzsun/classes/2018Fall_CS145/Slides/optimization.pdf) | ------ | ------ | ------ |
| Week 1 | 1/10 | Basics: Logistic Regression [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/02NaiveBayes_LR.pdf)| <ul><li>Naive Bayes: http://pages.cs.wisc.edu/~jerryzhu/cs769/nb.pdf </li><li>Newton Raphson Algorithm: https://www.stat.washington.edu/adobra/classes/536/Files/week1/newtonfull.pdf</li><li>Discriminative vs. generative: https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf</li></ul>| ------ | ------ | ------ |
| Week 2 | 1/15 | **Martin Luther King, Jr. holiday (No Class)** | -------- | ------ | ------ | ------ |
| Week 2 | 1/17 |Basics: K-Means, Gaussian Mixture Model [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/03kmeans_GMM.pdf)| <ul>Notes on mixture models and EM algorithm: <li>http://www.stat.cmu.edu/~cshalizi/350/lectures/29/lecture-29.pdf</li> <li>http://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/mixtureModels.pdf</li> </ul> | ------ | HW1 out | Team Sign-up Due |
| Week 3| 1/22 |Basics: Neural Networks, Deep Learning [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/04NN_DeepLearning.pdf) | <ul><li>3Blue1Brown NN series: https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi</li> <li> Deep Learning: http://neuralnetworksanddeeplearning.com/ http://www.deeplearningbook.org/ http://www.charuaggarwal.net/neural.htm http://d2l.ai/index.html </li></ul>| ------ | ------ | ------ |
| Week 3 | 1/24 | Text: Topic Model: PLSA [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/05TopicModels.pdf)| <ul><li>T. Hofmann. Probabilistic latent semantic indexing. Proceedings of the Twenty-Second Annual International SIGIR Conference, 1999.</li><li>David Blei, Andrew Ng, Michael Jordan, “Latent Dirichlet Allocation”, JMLR, 2003. http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf</li></ul> | ------ | HW1 Due; HW2 Out | ------ |
| Week 4 | 1/29 | Text: Topic Model: LDA and Variational Inference [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/05TopicModels.pdf)| <ul><li>David Blei, lecture notes on variational inference: https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf</li><li>David Blei, Andrew Ng, Michael Jordan, “Latent Dirichlet Allocation”, JMLR, 2003. http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf</li><li>Thomas L. Griffiths and Mark Steyvers, “Finding Scientific Topics”, PNAS, 2004. https://www.pnas.org/content/101/suppl_1/5228</li></ul> | ------ | ------ | ------ |
| Week 4 | 1/31 | Text: Word Embedding [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/06WordEmbedding.pdf)| <ul><li>Mikolov, T., Corrado, G., Chen, K., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. Proceedings of the International Conference on Learning Representations (ICLR 2013), 1–12.</li><li>Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. NIPS, 1–9.</li><li>Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 1532–1543.</li><li>Yoav Goldberg and Omer Levy (2014). Word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method. https://arxiv.org/pdf/1402.3722v1.pdf</li></ul> | ------ | HW2 Due; HW3 Out | ------ |
| Week 5 | 2/5 | Text: Transformers [[slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/07Transformers.pdf)| <ul><li>Course: https://deeplearning.cs.cmu.edu/F23/document/slides/lec19.transformersLLMs.pdf</li><li>A detailed introduction of Attention and Transformers: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html</li><li>Interactive visualization of encoder-only models (BERT): https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ?usp=sharing</li></ul>| ------ | ------ |
| Week 5 | 2/7 |  Graph: Spectral Clustering [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/08Graph_spectral.pdf) | <ul><li>Page et al. (1998) The PageRank Citation Ranking: Bringing Order to the Web. http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427</li><li>https://nlp.stanford.edu/projects/pagerank.shtml</li><li> A Tutorial on Spectral Clustering by U. Luxburg. https://arxiv.org/abs/0711.0189 </li></ul> | ------ | HW3 Due; HW4 Out | ------ |
| Week 6 | 2/12 | Graph: Label Propagation (slides as above)| <ul><li>Learning from Labeled and Unlabeled Data with Label Propagation,  by Xiaojin Zhu and Zoubin Ghahramani http://www.cs.cmu.edu/~zhuxj/pub/CMU-CALD-02-107.pdf</li><li>Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions,  by Xiaojin Zhu et al., ICML’03 https://www.aaai.org/Papers/ICML/2003/ICML03-118.pdf</li><li>Learning with Local and Global Consistency,  by Denny Zhou et al., NIPS’03 http://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf</li></ul> | ------ | ------ | Project Proposal Due |
| Week 6 | 2/14 | Graph: Shallow Embedding and KG Embedding [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/09Graph_embedding.pdf) | <ul><li>Bryan Perozzi, Rami Al-Rfou, Steven Skiena, DeepWalk: Online Learning of Social Representations, KDD’14</li><li>Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Qiaozhu Mei, LINE: Large-scale Information Network Embedding, WWW’15</li><li>Aditya Grover, Jure Leskovec, node2vec: Scalable Feature Learning for Networks, KDD’16</li></ul><ul><li>Bordes et al., Translating Embeddings for Modeling Multi-relational Data, NIPS 2013</li><li>Yang et al., Embedding entities and relations for learning and inference in knowledge bases, ICLR 2015</li><li>Zhiqing Sun, Zhihong Deng, Jian-Yun Nie, and Jian Tang. “RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.” ICLR’19.</li></ul>| ------ | HW4 Due (late due Friday); HW5 Out | ------ |
| Week 7 | 2/19 | **Presidents’ Day holiday (No Class)** | -------- | ------ | ------ | ------ |
| Week 7 | 2/21 | Midterm (in class)|  | ------ | ------ | ------ |
| Week 8 | 2/26 | Graph: Graph Neural Networks (slides as above)| <ul><li>Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017</li><li>Tutorial: http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf</li><li>Tutorial: http://tkipf.github.io/misc/SlidesCambridge.pdf</li></ul> | ------ | HW5 Due; HW6 Out  | ------ |
| Week 8 | 2/28 | RS: Collaborative Filtering, Matrix Factorization, and BPR [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/10Recommendation_MF.pdf) | <ul><li>[Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback, UAI 2009](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)</li></ul>  | ------ | ------ | ------ |
| Week 9 | 3/4 | RS: Factorization Machine and Neural Collaborative Filtering | -------- | ------ | HW6 Due | ------ |
| Week 9 | 3/6 | RS: Recommendation from Graph Perspective [[Slides]](http://web.cs.ucla.edu/~yzsun/classes/2024Winter_CS247/Slides/11Recommendation_Network.pdf) | -------- | ------ | ------ | ------ |
| Week 10 | 3/11 | Project Presentation | -------- | ------ | ------ | ------ |
| Week 10 | 3/13 | Project Presentation | -------- | ------ | ------ | Final Report / Code Due |
