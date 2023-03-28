1. **Why do we need deterministic relationships?**
   Many models require deterministic relationships for proper functioning. In this discussion, I will present examples of models that cannot be expressed without incorporating delta factors.

2. **Examining the delta node under a magnifying glass: What assumptions about inference can be made?**
   During the inference process, forward messages on inputs and backward messages on outputs are always accessible.

3. **Are there any issues?**
   Deterministic relationships can make our model non-conjugate. Unfortunately, even a simple graph with delta factors may not be possible to execute in a conjugate manner. For example, consider a simple normal-delta-normal model graph.

4. **Example of a simple normal-delta-normal model with equations:**
   Analytical derivation of marginals.

5. **A potential solution?**
   Utilize the backward message (Semih).

6. Generalize this solution for normals on Exponetial family
   Natural gradient ascent of the ELBO

7. **Forward message?**
   Employ the division trick. 

8. **CVI algorithm and CVMP algorithm for marginal calculations.**
   Summarize

9. **CVI algorithm and CVMP algorithm for message computations.**
   Summarize

10. **Comparison with linearization and models where linearization works well.**

11. **Comparison of results.**

12. **Models where linearization cannot be executed (and sampling is slow).**

13. **Comparison of results.**

14. **Issues encountered with the CVMP algorithm.**