---
layout: default
title: ADA STORY
cover: 'assets/images/cover1.jpg'
class: 'home-template'
navigation: True
logo: 'assets/images/Corcodile.png'
current: home
---

<header class="main-header {% if page.cover %}" style="background-image: url({{ site.baseurl }}{{ page.cover }}) {% else %}no-cover{% endif %}">
    <nav class="main-nav overlay clearfix">
        {% if page.logo %}<a class="blog-logo" href="{{ site.baseurl }}"><img src="{{ site.baseurl }}{{ page.logo }}" alt="Blog Logo" /></a>{% endif %}
        {% if page.navigation %}
            <a class="menu-button icon-menu" href="#"><span class="word">Menu</span></a>
        {% endif %}
    </nav>
    <div class="vertical">
        <div class="main-header-content inner">
            # Prologue
            ## Setting the scene...
            It’s a saturday night and you are scrolling through movies on Netflix...
            <!-- Continue the rest of the story in markdown format -->
        </div>
    </div>
    <a class="scroll-down icon-arrow-left" href="#content" data-offset="-45"><span class="hidden">Scroll Down</span></a>
</header>

<main id="content" class="content" role="main">
    <h1 class="page-title">{{ site.name }}</h1>
    <h2 class="page-description">{{ site.description }}</h2>

    ## Sections:
    - [Data](#data)
    - [Analysis](#analysis)
    - [Results](#results)
</main>
