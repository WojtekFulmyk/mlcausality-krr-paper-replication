<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[ ![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<!-- 
<br />
<div align="center">
  <a href="https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
-->

<h3 align="center">mlcausality-krr-paper-replication</h3>

  <p align="center">
    Replication code for the paper "Nonlinear Granger Causality using Kernel Ridge Regression"
    <br />
    <a href="https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication/issues">Report Bug</a>
    ·
    <a href="https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#downloading-the-replication-files">Downloading the Replication Files/a></li>
        <li><a href="#replicate-results">Replicate results/a></li>
      </ul>
    </li>
    <li><a href="#file-descriptions">File Descriptions</a></li>
      <ul>
        <li><a href="#files-from-the-lsngc-project">Files from the lsNGC project</a></li>
        <li><a href="#files-from-the-causal-ccm-project">Files from the causal_ccm project</a></li>
        <li><a href="#files-written-by-myself-with-the-help-of-some-code-from-the-spectral-connectivity-library">Files written by myself with the help of some code from the spectral-connectivity library</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- 
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
-->

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

This repository stores replication code for the paper "Nonlinear Granger Causality using Kernel Ridge Regression" that introduces and highlights the use if the <a href="https://github.com/WojtekFulmyk/mlcausality">mlcausality</a> package.

The replication files contained herein use functions from the following projects:
 <ul>
  <li><a href="https://github.com/Eden-Kramer-Lab/spectral_connectivity">spectral-connectivity</a></li>
  <li><a href="https://github.com/Large-scale-causality-inference/Large-scale-nonlinear-causality">lsNGC</a></li>
   <ul>
    <li>Wismüller, A., Dsouza, A.M., Vosoughi, M.A. et al. Large-scale nonlinear Granger causality for inferring directed dependence from short multivariate time-series data. Sci Rep 11, 7817 (2021). https://doi.org/10.1038/s41598-021-87316-6</li>
    <li>The public access to the paper is available here: https://www.nature.com/articles/s41598-021-87316-6</li>
   </ul>
  <li><a href="https://github.com/PrinceJavier/causal_ccm">causal_ccm</a></li>
   <ul>
    <li>Javier, P. J. E. (2021). causal-ccm a Python implementation of Convergent Cross Mapping (0.3.3) [Computer software].</li>
   </ul>
 </ul>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
### Prerequisites

Replication requires the installation of the following Python libraries:
* [NumPy](https://numpy.org)
* [SciPy](https://scipy.org)
* [pandas](https://pandas.pydata.org)
* [statsmodels](https://www.statsmodels.org)
* [scikit-learn](https://scikit-learn.org)
* [tqdm](https://github.com/tqdm/tqdm)
* [matplotlib](https://matplotlib.org/)
* [networkx](https://networkx.org/)
* [psutil](https://github.com/giampaolo/psutil)
* [tigramite](https://github.com/jakobrunge/tigramite)
* [mlcausality](https://github.com/WojtekFulmyk/mlcausality)

Please install libraries into your system before proceeding.

### Downloading the Replication Files

In order to replicate the results of the paper, you must download all of the files in this repository onto your local computer. You can do so, for instance, using the following command:

    git clone --depth 1 https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication.git

### Replicate results

You can replicate results by running all the codes in the `run_all` file on a terminal on your local machine.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- File Descriptions -->
## File Descriptions

### Files from the lsNGC project
The following repository files are taken from the <li><a href="https://github.com/Large-scale-causality-inference/Large-scale-nonlinear-causality">lsNGC</a</li> project. Note that these files were forked from the originals in order to perform some modifications. In particular, the `utils_mod.py` was modified to calculate p-values, and the `recovery_performance_mod.py` file was modified to ignore the diagonal of the adjacency matrix.

* `utils_mod.py`
* `recovery_performance_mod.py`
* `normalize_0_mean_1_std.py`
* `multivariate_split.py`
* `calc_f_stat.py`

### Files from the causal_ccm project
The following file was taken and modified from the <a href="https://github.com/PrinceJavier/causal_ccm">causal_ccm</a> project. In particular, an error involving underflow in exp was fixed.

* `causal_cmm.py`

### Files written by myself with the help of some code from the spectral-connectivity library
The following files were largely written by myself (Wojciech "Victor" Fulmyk) with the help of some code from the <a href="https://github.com/Eden-Kramer-Lab/spectral_connectivity">spectral-connectivity</a> project. In particular, <a href="https://github.com/Eden-Kramer-Lab/spectral_connectivity">spectral-connectivity</a> was used to generate the 5_linear network; see the `run_all_models.py` file for details. The relevant functions from the <a href="https://github.com/Eden-Kramer-Lab/spectral_connectivity">spectral-connectivity</a> package were forked herein as opposed to using the original package in order to ensure that the seed was set correctly for replication.

* `run_all_models.py`
* `make_all_multi_plots.py`
* `cao_min_embedding_dimension.py`
* `brier_loss.py`
* `run_all`
* `Readme.md` (this file)

<!-- LICENSE -->
## License

Distributed under the GPLv3 License because the  <a href="https://github.com/Eden-Kramer-Lab/spectral_connectivity">spectral-connectivity</a>, whose functions are used in this repository, are also released under GPLv3. See [LICENSE](https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication/blob/master/LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
<!-- 
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication](https://github.com/WojtekFulmyk/mlcausality-krr-paper-replication)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
-->


<!-- ACKNOWLEDGMENTS -->
<!-- 
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[contributors-url]: https://github.com/WojtekFulmyk/mlcausality/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[forks-url]: https://github.com/WojtekFulmyk/mlcausality/network/members
[stars-shield]: https://img.shields.io/github/stars/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[stars-url]: https://github.com/WojtekFulmyk/mlcausality/stargazers
[issues-shield]: https://img.shields.io/github/issues/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[issues-url]: https://github.com/WojtekFulmyk/mlcausality/issues
[license-shield]: https://img.shields.io/github/license/WojtekFulmyk/mlcausality.svg?style=for-the-badge
[license-url]: https://github.com/WojtekFulmyk/mlcausality/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

[NumPy-url]: https://numpy.org
[SciPy-url]: https://scipy.org
[pandas-url]: https://pandas.pydata.org
[statsmodels-url]: https://www.statsmodels.org
[scikit-learn-url]: https://scikit-learn.org
[XGBoost-url]: https://xgboost.readthedocs.io
[LightGBM-url]: https://lightgbm.readthedocs.io
[CatBoost-url]: https://catboost.ai
[cuML-url]: https://github.com/rapidsai/cuml


[NumPy-sheild]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[SciPy-sheild]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white
[Pandas-sheild]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[scikit-learn-shield]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white

